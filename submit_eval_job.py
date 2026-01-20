from utils import submit_job_bv
import os
os.environ["LMUDataRoot"] = "/proj/mmfm/LMUData"
os.environ["LMUData"] = "/proj/mmfm/LMUData"
# export LMUDataRoot="/proj/mmfm/LMUData"
# export LMUData="/proj/mmfm/LMUData"

# model = "granite_vision_3.3_2b"
model = "Phi-3.5-Vision"
datasets = [
    # "ChartCap",
    "ChartCapProbe",
    # "ChartCapSmall",
    # "DocVQA_VAL",
    # "ChartQA_TEST",
    # "AI2D_TEST",
    # "COCO_VAL",
    # "HallusionBench",
    # "RealWorldQA",
    # "POPE",
    # "BLINK",
    # "CCBench",
    # "MMStar",
    # "SEEDBench_IMG",
    # "SEEDBench2",
    # "SEEDBench2_Plus",
    # "MMMU_DEV_VAL",
    # "OCRVQA_TEST",
    # "OCRVQA_TESTCORE"
    # "TextVQA_VAL",
    # "LLaVABench",           
    # "MathVista_MINI",
    # "OCRBench",
    # "InfoVQA_VAL"
]

# command = f'pyutils-run run.py --model {model} --data {" ".join(datasets)}'
command = f'python run.py --model {model} --data {" ".join(datasets)}'
print("submitting")
print_only = False
outputs = submit_job_bv(
    cmd=command,
    gpu_num=8,
    node_num=1,
    name="gv_eval_vlm",
    group="grp_preemptable",
    print_only=print_only,
    queue="preemptable"
)
print(outputs)
