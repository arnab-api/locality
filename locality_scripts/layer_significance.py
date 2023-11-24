import argparse
import logging
from typing import Union

import baukit
import numpy as np
from tqdm import tqdm

import locality.metric as metric
from causal_trace.utils import ModelandTokenizer
from dsets.counterfact import CounterFactDataset
from locality.dataset import (
    QA_Sample,
    VariableBindingFactRecallDataset,
    generate_synthetic_dataset,
)
from locality.functional import (
    filter_samples_by_model_knowledge,
    find_token_range,
    get_h,
    patch_output,
    predict_next_token,
)
from locality.utils import experiment_utils, logging_utils
from locality.utils.dataclasses import (
    ExperimentResults,
    LayerPatchingEfficacy,
    PatchingResults_for_one_pair,
    PatchingTrialResult,
    PredictedToken,
)

logger = logging.getLogger(__name__)


def get_edit_target(qa_samples: list[QA_Sample]) -> dict[str, QA_Sample]:
    """Given a list of (subject, object) pairs, return a mapping from each pair to a
    different pair that differs in only one element.
    """
    edit_target = {}
    for idx in range(len(qa_samples)):
        jdx = idx
        while qa_samples[idx].subject == qa_samples[jdx].subject:
            jdx = np.random.choice(len(qa_samples))
        edit_target[qa_samples[idx].subject] = qa_samples[jdx]
    return edit_target


def patch_individual_layers_for_single_edit(
    mt: ModelandTokenizer,
    layers: list[int],
    source_QA: QA_Sample,
    edit_QA: QA_Sample,
    query: str,
) -> PatchingResults_for_one_pair:
    # TODO: Support for multiple edits
    # ! Multiple edit acting weird. Could not find out the bug.
    edit_h = get_h(
        mt=mt,
        prompt=query.replace(source_QA.subject, edit_QA.subject),
        subject=edit_QA.subject,
        layers=[mt.layer_name_format.format(layer_idx) for layer_idx in layers],
    )

    tokenized = mt.tokenizer(
        query, return_offsets_mapping=True, return_tensors="pt"
    ).to(mt.model.device)
    offset_mapping = tokenized.pop("offset_mapping")[0]

    subject_start, subject_end = find_token_range(
        query, source_QA.subject, tokenizer=mt.tokenizer, offset_mapping=offset_mapping
    )

    subj_last_idx = subject_end - 1
    edit_rank_after_patching: dict[int, tuple[int, PredictedToken]] = {}
    predictions: dict[int, list[PredictedToken]] = {}
    edit_token = mt.tokenizer.decode(tokenized["input_ids"][0][subj_last_idx])

    logger.debug("=" * 100)
    logger.debug(
        f"({source_QA.subject}, {source_QA.answer}) => ({edit_QA.subject}, {edit_QA.answer}) | edit_idx={subj_last_idx}[{edit_token}]"
    )

    for layer_idx in layers:
        layer_name = mt.layer_name_format.format(layer_idx)
        with baukit.Trace(
            module=mt.model,
            layer=layer_name,
            edit_output=patch_output(
                patch_layer=layer_name,
                patch_idx=subj_last_idx,
                patching_vector=edit_h[layer_name],
            ),
        ):
            preds, edit_answer_rank = predict_next_token(
                mt=mt, prompt=query, token_of_interest=edit_QA.answer
            )
        predictions[layer_idx] = preds[0]
        edit_rank_after_patching[layer_idx] = edit_answer_rank[0]
        logger.debug(
            f"Layer {layer_idx} => rank({edit_QA.answer})={edit_answer_rank[0][0]} [{edit_answer_rank[0][1]}]  | preds={', '.join(str(p) for p in preds[0])}"
        )
    logger.debug("-" * 100)

    return PatchingResults_for_one_pair(
        source_QA=source_QA,
        edit_QA=edit_QA,
        edit_index=subj_last_idx,
        edit_token=mt.tokenizer.decode(tokenized["input_ids"][0][subj_last_idx]),
        predictions_after_patching=predictions,
        rank_edit_ans_after_patching=edit_rank_after_patching,
    )


def measure_patching_effect_on_each_layer(
    mt: ModelandTokenizer,
    layers: list[int],
    dataset: VariableBindingFactRecallDataset,
    edit_target: dict[str, QA_Sample],
):
    lawerwise_predictions_after_patching: dict[int, list[list[str]]] = {
        layer_idx: [] for layer_idx in layers
    }
    layerwise_edit_answer_rank_after_patching: dict[int, list[int]] = {
        layer_idx: [] for layer_idx in layers
    }
    target_answers: list[str] = []

    track_edit_results = []
    for idx in tqdm(range(len(dataset))):
        source_QA = dataset.qa_samples[idx]
        edit_QA = edit_target[source_QA.subject]
        query = dataset[idx][0]
        edit_result = patch_individual_layers_for_single_edit(
            mt=mt,
            layers=layers,
            source_QA=source_QA,
            edit_QA=edit_QA,
            query=query,
        )
        track_edit_results.append(edit_result)

        target_answers.append(edit_QA.answer)
        for layer_idx in layers:
            lawerwise_predictions_after_patching[layer_idx].append(
                [
                    pred.token
                    for pred in edit_result.predictions_after_patching[layer_idx]
                ]
            )
            layerwise_edit_answer_rank_after_patching[layer_idx].append(
                edit_result.rank_edit_ans_after_patching[layer_idx]
            )

    logger.info("=" * 100)
    patching_effect: list[LayerPatchingEfficacy] = []
    for layer_idx in layers:
        layer_recall = metric.recall(
            predictions=lawerwise_predictions_after_patching[layer_idx],
            targets=target_answers,
        )
        reciprocal_rank = metric.reciprocal_rank(
            [rank[0] for rank in layerwise_edit_answer_rank_after_patching[layer_idx]]
        )
        logger.info(f"Layer {layer_idx} => {layer_recall=}, {reciprocal_rank=}")
        patching_effect.append(
            LayerPatchingEfficacy(
                layer_idx=layer_idx,
                recall=layer_recall,
                reciprocal_rank=reciprocal_rank,
            )
        )
    logger.info("=" * 100)
    return patching_effect, track_edit_results


def run_trial(
    mt,
    layers: list[int],
    subj_obj_mapping: list[tuple[str, str]],
    num_options: int = 3,  # number of options to choose from in the query
    num_icl: int = 5,  # number of few-shot examples to contextualize
    dataset_size=100,
    variable_binding_template=" {} is visiting {}",
    query_template=" {} is in {}.",
    filter_prompt_template=" {} is located in the country of",
):
    # filter out samples that the model knows
    icl_examples = [
        subj_obj_mapping[k]
        for k in np.random.choice(len(subj_obj_mapping), size=num_icl, replace=False)
    ]
    filtered_subj_obj_mapping = filter_samples_by_model_knowledge(
        mt,
        subj_obj_mapping,
        prompt_template=filter_prompt_template,
        icl_examples=icl_examples,
    )
    logger.info(f"Filtered to {len(filtered_subj_obj_mapping)} samples")

    # generate synthetic dataset
    dataset = generate_synthetic_dataset(
        filtered_subj_obj_mapping,
        variable_binding_template=variable_binding_template,
        query_template=query_template,
        num_options=num_options,
        num_icl=num_icl,
        batch_size=dataset_size,
    )

    edit_target = get_edit_target(dataset.qa_samples)

    patching_effect, track_edit_results = measure_patching_effect_on_each_layer(
        mt=mt,
        layers=layers,
        dataset=dataset,
        edit_target=edit_target,
    )
    return PatchingTrialResult(
        few_shot_demonstration=icl_examples,
        layer_patching_effecacy=patching_effect,
        patching_results=track_edit_results,
    )


def layer_significance_experiment(
    model_path: str = "meta-llama/Llama-2-7b-hf",
    layers: list[int] = list(range(32)),
    num_trials=10,
    counterfact_relation_id="P17",
    dataset_size=500,
    num_options=3,
    num_icl=5,
    variable_binding_template=" {} is visiting {}",
    query_template=" {} is in {}",
    filter_prompt_template=" {} is located in the country of",
    results_dir: str = "results",
):
    mt = ModelandTokenizer(model_path=model_path)

    counterfact = counterfact = CounterFactDataset(data_dir="counterfact")
    relation_filter = [
        d
        for d in counterfact
        if d["requested_rewrite"]["relation_id"] == counterfact_relation_id
    ]
    relation_subj_obj_mapping = [
        (
            d["requested_rewrite"]["subject"],
            d["requested_rewrite"]["target_true"]["str"],
        )
        for d in relation_filter
    ]

    experiment_results = ExperimentResults(
        experiment_specific_args=dict(
            model_path=model_path,
            counterfact_relation_id=counterfact_relation_id,
            dataset_size=dataset_size,
            num_options=num_options,
            num_icl=num_icl,
            variable_binding_template=variable_binding_template,
            query_template=query_template,
        ),
        trial_results=[],
    )

    for trial_no in range(num_trials):
        logger.info(f"################## Trial {trial_no + 1} ##################")

        trial = run_trial(
            mt,
            layers=layers,
            subj_obj_mapping=relation_subj_obj_mapping,
            variable_binding_template=variable_binding_template,
            query_template=query_template,
            filter_prompt_template=filter_prompt_template,
            num_options=num_options,
            num_icl=num_icl,
            dataset_size=dataset_size,
        )

        experiment_results.trial_results.append(trial)
        experiment_utils.save_results_file(
            results_dir=results_dir,
            name=f'{model_path.split("/")[-1]}_{counterfact_relation_id}_layer_edit_efficacy',
            results=experiment_results,
        )
        logger.info(f"###############################################\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="sweep over hyperparameters")
    logging_utils.add_logging_args(parser)
    experiment_utils.add_experiment_args(parser)

    parser.add_argument(
        "--model-path",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
    )

    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=list(range(32)),
        help="layers to patch",
    )

    parser.add_argument(
        "--n-trials",
        type=int,
        default=3,
        help="number of trials per relation",
    )

    parser.add_argument(
        "--relation-id",
        type=str,
        default="P17",
        help="counterfact relation id",
    )

    parser.add_argument(
        "--dataset-size",
        type=int,
        default=100,
        help="number of samples per trial",
    )

    parser.add_argument(
        "--num-options",
        type=int,
        default=3,
        help="number of options to choose from in the query",
    )

    parser.add_argument(
        "--num-icl",
        type=int,
        default=5,
        help="number of few-shot examples to contextualize",
    )

    parser.add_argument(
        "--variable-binding-template",
        type=str,
        default=" {} is visiting {}",
    )

    parser.add_argument(
        "--query-template",
        type=str,
        default=" {} is in {}",
    )

    args = parser.parse_args()
    logging_utils.configure(args)
    experiment = experiment_utils.setup_experiment(args)

    logger.info(args)

    layer_significance_experiment(
        model_path=args.model_path,
        layers=args.layers,
        num_trials=args.n_trials,
        counterfact_relation_id=args.relation_id,
        dataset_size=args.dataset_size,
        num_options=args.num_options,
        num_icl=args.num_icl,
        variable_binding_template=args.variable_binding_template,
        query_template=args.query_template,
        results_dir=experiment.results_dir,
    )
