# KARMA-MV
A Benchmark for Causal Question Answering on Music Videos
The repo comsists of the codes important to run the modles
The [KARMA-MV dataset] [1] KARMA-MV dataset contains MCQs based on Music Video-Causality [2] There are 2686 Music videos from which we filtered some relevant transition scene-pairs and framed questions on the causality in Music and video. Mainly how muisc changes with video [3] Each transition pair has 3 questions each of the types (a) Evidence Reasoning (b) Predictive type (c) Counterfactual [4] The MCQs are generated using Qwen-2.5-7B-instruct LLM. [5] Some sample clips have been uploaded in <.>

The dataset is in HuggingFace: https://huggingface.co/datasets/amaai-lab/KARMA-MV

If you use this dataset, please cite the paper in which it is presented: Archishman Ghosh, Abhinaba Roy, Dorien Herremans, 2026, KARMA-MV: A BENCHMARK FOR CAUSAL QUESTION ANSWERING ON MUSIC VIDEOS.
@article{A_Ghosh2026,
  author    = {Archishman Ghosh and Abhinaba Roy and Dorien Herremans},
  title     = {KARMA-MV: A BENCHMARK FOR CAUSAL QUESTION ANSWERING
ON MUSIC VIDEOS},
  year      = {2026},
  journal   = {arXiv:preprint}
}
