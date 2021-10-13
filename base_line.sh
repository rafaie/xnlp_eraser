export MODEL_NAME=bert-base-uncased
export MODEL_TYPE=fine_tune_baseline_sp
export TRAIN_DATA_PATH=./data/esnli/train.jsonl
export VALIDATION_DATA_PATH=./data/esnli/val.jsonl
export TEST_DATA_PATH=./data/esnli/test.jsonl
export LABELS=neutral,entailment,contradiction
export SEED_NUMBER=0
export CUDA_DEVICE=0
export SERIALIZATION_DIR=output/$MODEL_TYPE

#allennlp train experiment_1/conf/baseline_linear.jsonnet \
#	    --s output/$MODEL_TYPE \
#	        --include-package experiment_1 

allennlp predict \
	--predictor rationale_predictor \
	--include-package experiment_1 \
	--silent \
	--batch-size 1 \
	--use-dataset-reader \
	--dataset-reader-choice validation \
	--output-file output/$MODEL_TYPE/pred.jsonl \
	output/$MODEL_TYPE/model.tar.gz \
	./data/esnli/test.jsonl


python3 eraserbenchmark/rationale_benchmark/metrics.py \
	--data_dir data/esnli \
	--split test \
	--results output/$MODEL_TYPE/pred.jsonl \
	--score_file output/$MODEL_TYPE/test_scores.json
	

