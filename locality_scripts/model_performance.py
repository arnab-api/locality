import argparse
import logging

import numpy as np
from tqdm import tqdm

import locality.metric as metric
from causal_trace.utils import ModelandTokenizer
from dsets.counterfact import CounterFactDataset
from locality.dataset import generate_synthetic_dataset
from locality.functional import filter_samples_by_model_knowledge, predict_next_token
from locality.utils import experiment_utils, logging_utils
from locality.utils.dataclasses import ExperimentResults, SampleResult, TrialResult

logger = logging.getLogger(__name__)


def run_trial(
    mt,
    subj_obj_mapping: list[tuple[str, str]],
    variable_binding_template=" {} is visiting {}",
    query_template=" {} is in {}.",
    num_options: int = 3,  # number of options to choose from in the query
    num_icl: int = 5,  # number of few-shot examples to contextualize
    dataset_size=100,
):
    # filter out samples that the model knows
    icl_examples = [
        subj_obj_mapping[k]
        for k in np.random.choice(len(subj_obj_mapping), size=num_icl, replace=False)
    ]
    filtered_subj_obj_mapping = filter_samples_by_model_knowledge(
        mt,
        subj_obj_mapping,
        prompt_template=" {} is located in the country of",
        icl_examples=icl_examples,
    )

    # generate synthetic dataset
    dataset = generate_synthetic_dataset(
        filtered_subj_obj_mapping,
        variable_binding_template=variable_binding_template,
        query_template=query_template,
        num_options=num_options,
        num_icl=num_icl,
        batch_size=dataset_size,
    )

    sample_results = []
    predictions = []
    targets = []
    for query, answer in tqdm(dataset):
        pred = predict_next_token(mt, query, k=3)[0]
        sample_results.append(
            SampleResult(
                query=query,
                answer=answer,
                prediction=pred,
            )
        )
        predictions.append([p.token for p in pred])
        targets.append(answer)

    return TrialResult(
        few_shot_demonstration=icl_examples,
        samples=sample_results,
        recall=metric.recall(predictions, targets),
    )


def measure_model_performance(
    model_path: str = "meta-llama/Llama-2-7b-hf",
    num_trials=10,
    counterfact_relation_id="P17",
    dataset_size=100,
    num_options=3,
    num_icl=5,
    variable_binding_template=" {} is visiting {}",
    query_template=" {} is in {}",
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
    ][:100]

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
            subj_obj_mapping=relation_subj_obj_mapping,
            variable_binding_template=variable_binding_template,
            query_template=query_template,
            num_options=num_options,
            num_icl=num_icl,
        )
        logger.debug("-" * 50)
        logger.info(f"Recall: {trial.recall}")
        logger.debug("-" * 50)

        experiment_results.trial_results.append(trial)
        experiment_utils.save_results_file(
            results_dir=results_dir,
            name=f'{model_path.split("/")[-1]}_{counterfact_relation_id}_performance',
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
    experiment_utils.setup_experiment(args)

    logger.info(args)

    measure_model_performance(
        model_path=args.model_path,
        num_trials=args.n_trials,
        counterfact_relation_id=args.relation_id,
        dataset_size=args.dataset_size,
        num_options=args.num_options,
        num_icl=args.num_icl,
        variable_binding_template=args.variable_binding_template,
        query_template=args.query_template,
        results_dir=args.results_dir,
    )
