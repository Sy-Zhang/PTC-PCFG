echo '0.1Tars-CPCFGs'
python -m torch.distributed.launch --nproc_per_node=1 --master_port 1114 parsing/run.py --cfg experiments/HowTo100M/0.1Tars-CPCFGs.yaml --test_mode --split test --checkpoint checkpoints/HowTo100M/0.1Tars-CPCFGs/seed-1110/epoch0000.pkl
python -m torch.distributed.launch --nproc_per_node=1 --master_port 1114 parsing/run.py --cfg experiments/HowTo100M/0.1Tars-CPCFGs.yaml --test_mode --split test --checkpoint checkpoints/HowTo100M/0.1Tars-CPCFGs/seed-1117/epoch0000.pkl
python -m torch.distributed.launch --nproc_per_node=1 --master_port 1114 parsing/run.py --cfg experiments/HowTo100M/0.1Tars-CPCFGs.yaml --test_mode --split test --checkpoint checkpoints/HowTo100M/0.1Tars-CPCFGs/seed-1119/epoch0000.pkl
python -m torch.distributed.launch --nproc_per_node=1 --master_port 1114 parsing/run.py --cfg experiments/HowTo100M/0.1Tars-CPCFGs.yaml --test_mode --split test --checkpoint checkpoints/HowTo100M/0.1Tars-CPCFGs/seed-1120/epoch0000.pkl
python -m torch.distributed.launch --nproc_per_node=1 --master_port 1114 parsing/run.py --cfg experiments/HowTo100M/0.1Tars-CPCFGs.yaml --test_mode --split test --checkpoint checkpoints/HowTo100M/0.1Tars-CPCFGs/seed-1121/epoch0000.pkl

echo '1Tars-CPCFGs'
python -m torch.distributed.launch --nproc_per_node=1 --master_port 1114 parsing/run.py --cfg experiments/HowTo100M/1Tars-CPCFGs.yaml --test_mode --split test --checkpoint checkpoints/HowTo100M/1Tars-CPCFGs/seed-1110/epoch0000.pkl
python -m torch.distributed.launch --nproc_per_node=1 --master_port 1114 parsing/run.py --cfg experiments/HowTo100M/1Tars-CPCFGs.yaml --test_mode --split test --checkpoint checkpoints/HowTo100M/1Tars-CPCFGs/seed-1117/epoch0000.pkl
python -m torch.distributed.launch --nproc_per_node=1 --master_port 1114 parsing/run.py --cfg experiments/HowTo100M/1Tars-CPCFGs.yaml --test_mode --split test --checkpoint checkpoints/HowTo100M/1Tars-CPCFGs/seed-1119/epoch0000.pkl
python -m torch.distributed.launch --nproc_per_node=1 --master_port 1114 parsing/run.py --cfg experiments/HowTo100M/1Tars-CPCFGs.yaml --test_mode --split test --checkpoint checkpoints/HowTo100M/1Tars-CPCFGs/seed-1120/epoch0000.pkl
python -m torch.distributed.launch --nproc_per_node=1 --master_port 1114 parsing/run.py --cfg experiments/HowTo100M/1Tars-CPCFGs.yaml --test_mode --split test --checkpoint checkpoints/HowTo100M/1Tars-CPCFGs/seed-1121/epoch0000.pkl

echo '2Tars-CPCFGs'
python -m torch.distributed.launch --nproc_per_node=1 --master_port 1114 parsing/run.py --cfg experiments/HowTo100M/2Tars-CPCFGs.yaml --test_mode --split test --checkpoint checkpoints/HowTo100M/2Tars-CPCFGs/seed-1110/epoch0000.pkl
python -m torch.distributed.launch --nproc_per_node=1 --master_port 1114 parsing/run.py --cfg experiments/HowTo100M/2Tars-CPCFGs.yaml --test_mode --split test --checkpoint checkpoints/HowTo100M/2Tars-CPCFGs/seed-1117/epoch0000.pkl
python -m torch.distributed.launch --nproc_per_node=1 --master_port 1114 parsing/run.py --cfg experiments/HowTo100M/2Tars-CPCFGs.yaml --test_mode --split test --checkpoint checkpoints/HowTo100M/2Tars-CPCFGs/seed-1119/epoch0000.pkl
python -m torch.distributed.launch --nproc_per_node=1 --master_port 1114 parsing/run.py --cfg experiments/HowTo100M/2Tars-CPCFGs.yaml --test_mode --split test --checkpoint checkpoints/HowTo100M/2Tars-CPCFGs/seed-1120/epoch0000.pkl
python -m torch.distributed.launch --nproc_per_node=1 --master_port 1114 parsing/run.py --cfg experiments/HowTo100M/2Tars-CPCFGs.yaml --test_mode --split test --checkpoint checkpoints/HowTo100M/2Tars-CPCFGs/seed-1121/epoch0000.pkl

echo '4Tars-CPCFGs'
python -m torch.distributed.launch --nproc_per_node=1 --master_port 1114 parsing/run.py --cfg experiments/HowTo100M/4Tars-CPCFGs.yaml --test_mode --split test --checkpoint checkpoints/HowTo100M/4Tars-CPCFGs/seed-1110/epoch0000.pkl
python -m torch.distributed.launch --nproc_per_node=1 --master_port 1114 parsing/run.py --cfg experiments/HowTo100M/4Tars-CPCFGs.yaml --test_mode --split test --checkpoint checkpoints/HowTo100M/4Tars-CPCFGs/seed-1117/epoch0000.pkl
python -m torch.distributed.launch --nproc_per_node=1 --master_port 1114 parsing/run.py --cfg experiments/HowTo100M/4Tars-CPCFGs.yaml --test_mode --split test --checkpoint checkpoints/HowTo100M/4Tars-CPCFGs/seed-1119/epoch0000.pkl
python -m torch.distributed.launch --nproc_per_node=1 --master_port 1114 parsing/run.py --cfg experiments/HowTo100M/4Tars-CPCFGs.yaml --test_mode --split test --checkpoint checkpoints/HowTo100M/4Tars-CPCFGs/seed-1120/epoch0000.pkl
python -m torch.distributed.launch --nproc_per_node=1 --master_port 1114 parsing/run.py --cfg experiments/HowTo100M/4Tars-CPCFGs.yaml --test_mode --split test --checkpoint checkpoints/HowTo100M/4Tars-CPCFGs/seed-1121/epoch0000.pkl

echo '8Tars-CPCFGs'
python -m torch.distributed.launch --nproc_per_node=1 --master_port 1114 parsing/run.py --cfg experiments/HowTo100M/8Tars-CPCFGs.yaml --test_mode --split test --checkpoint checkpoints/HowTo100M/8Tars-CPCFGs/seed-1110/epoch0000.pkl
python -m torch.distributed.launch --nproc_per_node=1 --master_port 1114 parsing/run.py --cfg experiments/HowTo100M/8Tars-CPCFGs.yaml --test_mode --split test --checkpoint checkpoints/HowTo100M/8Tars-CPCFGs/seed-1117/epoch0000.pkl
python -m torch.distributed.launch --nproc_per_node=1 --master_port 1114 parsing/run.py --cfg experiments/HowTo100M/8Tars-CPCFGs.yaml --test_mode --split test --checkpoint checkpoints/HowTo100M/8Tars-CPCFGs/seed-1119/epoch0000.pkl
python -m torch.distributed.launch --nproc_per_node=1 --master_port 1114 parsing/run.py --cfg experiments/HowTo100M/8Tars-CPCFGs.yaml --test_mode --split test --checkpoint checkpoints/HowTo100M/8Tars-CPCFGs/seed-1120/epoch0000.pkl
python -m torch.distributed.launch --nproc_per_node=1 --master_port 1114 parsing/run.py --cfg experiments/HowTo100M/8Tars-CPCFGs.yaml --test_mode --split test --checkpoint checkpoints/HowTo100M/8Tars-CPCFGs/seed-1121/epoch0000.pkl

echo '0.1Tars-MMCPCFGs'
python -m torch.distributed.launch --nproc_per_node=1 --master_port 1114 parsing/run.py --cfg experiments/HowTo100M/0.1Tars-MMCPCFGs.yaml --test_mode --split test --checkpoint checkpoints/HowTo100M/0.1Tars-MMCPCFGs/seed-1110/epoch0000.pkl
python -m torch.distributed.launch --nproc_per_node=1 --master_port 1114 parsing/run.py --cfg experiments/HowTo100M/0.1Tars-MMCPCFGs.yaml --test_mode --split test --checkpoint checkpoints/HowTo100M/0.1Tars-MMCPCFGs/seed-1117/epoch0000.pkl
python -m torch.distributed.launch --nproc_per_node=1 --master_port 1114 parsing/run.py --cfg experiments/HowTo100M/0.1Tars-MMCPCFGs.yaml --test_mode --split test --checkpoint checkpoints/HowTo100M/0.1Tars-MMCPCFGs/seed-1119/epoch0000.pkl
python -m torch.distributed.launch --nproc_per_node=1 --master_port 1114 parsing/run.py --cfg experiments/HowTo100M/0.1Tars-MMCPCFGs.yaml --test_mode --split test --checkpoint checkpoints/HowTo100M/0.1Tars-MMCPCFGs/seed-1120/epoch0000.pkl
python -m torch.distributed.launch --nproc_per_node=1 --master_port 1114 parsing/run.py --cfg experiments/HowTo100M/0.1Tars-MMCPCFGs.yaml --test_mode --split test --checkpoint checkpoints/HowTo100M/0.1Tars-MMCPCFGs/seed-1121/epoch0000.pkl

echo '1Tars-MMCPCFGs'
python -m torch.distributed.launch --nproc_per_node=1 --master_port 1114 parsing/run.py --cfg experiments/HowTo100M/1Tars-MMCPCFGs.yaml --test_mode --split test --checkpoint checkpoints/HowTo100M/1Tars-MMCPCFGs/seed-1110/epoch0000.pkl
python -m torch.distributed.launch --nproc_per_node=1 --master_port 1114 parsing/run.py --cfg experiments/HowTo100M/1Tars-MMCPCFGs.yaml --test_mode --split test --checkpoint checkpoints/HowTo100M/1Tars-MMCPCFGs/seed-1117/epoch0000.pkl
python -m torch.distributed.launch --nproc_per_node=1 --master_port 1114 parsing/run.py --cfg experiments/HowTo100M/1Tars-MMCPCFGs.yaml --test_mode --split test --checkpoint checkpoints/HowTo100M/1Tars-MMCPCFGs/seed-1119/epoch0000.pkl
python -m torch.distributed.launch --nproc_per_node=1 --master_port 1114 parsing/run.py --cfg experiments/HowTo100M/1Tars-MMCPCFGs.yaml --test_mode --split test --checkpoint checkpoints/HowTo100M/1Tars-MMCPCFGs/seed-1120/epoch0000.pkl
python -m torch.distributed.launch --nproc_per_node=1 --master_port 1114 parsing/run.py --cfg experiments/HowTo100M/1Tars-MMCPCFGs.yaml --test_mode --split test --checkpoint checkpoints/HowTo100M/1Tars-MMCPCFGs/seed-1121/epoch0000.pkl

echo '2Tars-MMCPCFGs'
python -m torch.distributed.launch --nproc_per_node=1 --master_port 1114 parsing/run.py --cfg experiments/HowTo100M/2Tars-MMCPCFGs.yaml --test_mode --split test --checkpoint checkpoints/HowTo100M/2Tars-MMCPCFGs/seed-1110/epoch0000.pkl
python -m torch.distributed.launch --nproc_per_node=1 --master_port 1114 parsing/run.py --cfg experiments/HowTo100M/2Tars-MMCPCFGs.yaml --test_mode --split test --checkpoint checkpoints/HowTo100M/2Tars-MMCPCFGs/seed-1117/epoch0000.pkl
python -m torch.distributed.launch --nproc_per_node=1 --master_port 1114 parsing/run.py --cfg experiments/HowTo100M/2Tars-MMCPCFGs.yaml --test_mode --split test --checkpoint checkpoints/HowTo100M/2Tars-MMCPCFGs/seed-1119/epoch0000.pkl
python -m torch.distributed.launch --nproc_per_node=1 --master_port 1114 parsing/run.py --cfg experiments/HowTo100M/2Tars-MMCPCFGs.yaml --test_mode --split test --checkpoint checkpoints/HowTo100M/2Tars-MMCPCFGs/seed-1120/epoch0000.pkl
python -m torch.distributed.launch --nproc_per_node=1 --master_port 1114 parsing/run.py --cfg experiments/HowTo100M/2Tars-MMCPCFGs.yaml --test_mode --split test --checkpoint checkpoints/HowTo100M/2Tars-MMCPCFGs/seed-1121/epoch0000.pkl

echo '4Tars-MMCPCFGs'
python -m torch.distributed.launch --nproc_per_node=1 --master_port 1114 parsing/run.py --cfg experiments/HowTo100M/4Tars-MMCPCFGs.yaml --test_mode --split test --checkpoint checkpoints/HowTo100M/4Tars-MMCPCFGs/seed-1110/epoch0000.pkl
python -m torch.distributed.launch --nproc_per_node=1 --master_port 1114 parsing/run.py --cfg experiments/HowTo100M/4Tars-MMCPCFGs.yaml --test_mode --split test --checkpoint checkpoints/HowTo100M/4Tars-MMCPCFGs/seed-1117/epoch0000.pkl
python -m torch.distributed.launch --nproc_per_node=1 --master_port 1114 parsing/run.py --cfg experiments/HowTo100M/4Tars-MMCPCFGs.yaml --test_mode --split test --checkpoint checkpoints/HowTo100M/4Tars-MMCPCFGs/seed-1119/epoch0000.pkl
python -m torch.distributed.launch --nproc_per_node=1 --master_port 1114 parsing/run.py --cfg experiments/HowTo100M/4Tars-MMCPCFGs.yaml --test_mode --split test --checkpoint checkpoints/HowTo100M/4Tars-MMCPCFGs/seed-1120/epoch0000.pkl
python -m torch.distributed.launch --nproc_per_node=1 --master_port 1114 parsing/run.py --cfg experiments/HowTo100M/4Tars-MMCPCFGs.yaml --test_mode --split test --checkpoint checkpoints/HowTo100M/4Tars-MMCPCFGs/seed-1121/epoch0000.pkl

echo '8Tars-MMCPCFGs'
python -m torch.distributed.launch --nproc_per_node=1 --master_port 1114 parsing/run.py --cfg experiments/HowTo100M/8Tars-MMCPCFGs.yaml --test_mode --split test --checkpoint checkpoints/HowTo100M/8Tars-MMCPCFGs/seed-1110/epoch0000.pkl
python -m torch.distributed.launch --nproc_per_node=1 --master_port 1114 parsing/run.py --cfg experiments/HowTo100M/8Tars-MMCPCFGs.yaml --test_mode --split test --checkpoint checkpoints/HowTo100M/8Tars-MMCPCFGs/seed-1117/epoch0000.pkl
python -m torch.distributed.launch --nproc_per_node=1 --master_port 1114 parsing/run.py --cfg experiments/HowTo100M/8Tars-MMCPCFGs.yaml --test_mode --split test --checkpoint checkpoints/HowTo100M/8Tars-MMCPCFGs/seed-1119/epoch0000.pkl
python -m torch.distributed.launch --nproc_per_node=1 --master_port 1114 parsing/run.py --cfg experiments/HowTo100M/8Tars-MMCPCFGs.yaml --test_mode --split test --checkpoint checkpoints/HowTo100M/8Tars-MMCPCFGs/seed-1120/epoch0000.pkl
python -m torch.distributed.launch --nproc_per_node=1 --master_port 1114 parsing/run.py --cfg experiments/HowTo100M/8Tars-MMCPCFGs.yaml --test_mode --split test --checkpoint checkpoints/HowTo100M/8Tars-MMCPCFGs/seed-1121/epoch0000.pkl


echo '0.1Tars-PTCPCFGs'
python -m torch.distributed.launch --nproc_per_node=1 --master_port 1114 parsing/run.py --cfg experiments/HowTo100M/0.1Tars-PTCPCFGs.yaml --test_mode --split test --checkpoint checkpoints/HowTo100M/0.1Tars-PTCPCFGs/seed-1110/epoch0000.pkl
python -m torch.distributed.launch --nproc_per_node=1 --master_port 1114 parsing/run.py --cfg experiments/HowTo100M/0.1Tars-PTCPCFGs.yaml --test_mode --split test --checkpoint checkpoints/HowTo100M/0.1Tars-PTCPCFGs/seed-1117/epoch0000.pkl
python -m torch.distributed.launch --nproc_per_node=1 --master_port 1114 parsing/run.py --cfg experiments/HowTo100M/0.1Tars-PTCPCFGs.yaml --test_mode --split test --checkpoint checkpoints/HowTo100M/0.1Tars-PTCPCFGs/seed-1119/epoch0000.pkl
python -m torch.distributed.launch --nproc_per_node=1 --master_port 1114 parsing/run.py --cfg experiments/HowTo100M/0.1Tars-PTCPCFGs.yaml --test_mode --split test --checkpoint checkpoints/HowTo100M/0.1Tars-PTCPCFGs/seed-1120/epoch0000.pkl
python -m torch.distributed.launch --nproc_per_node=1 --master_port 1114 parsing/run.py --cfg experiments/HowTo100M/0.1Tars-PTCPCFGs.yaml --test_mode --split test --checkpoint checkpoints/HowTo100M/0.1Tars-PTCPCFGs/seed-1121/epoch0000.pkl

echo '1Tars-PTCPCFGs'
python -m torch.distributed.launch --nproc_per_node=1 --master_port 1114 parsing/run.py --cfg experiments/HowTo100M/1Tars-PTCPCFGs.yaml --test_mode --split test --checkpoint checkpoints/HowTo100M/1Tars-PTCPCFGs/seed-1110/epoch0000.pkl
python -m torch.distributed.launch --nproc_per_node=1 --master_port 1114 parsing/run.py --cfg experiments/HowTo100M/1Tars-PTCPCFGs.yaml --test_mode --split test --checkpoint checkpoints/HowTo100M/1Tars-PTCPCFGs/seed-1117/epoch0000.pkl
python -m torch.distributed.launch --nproc_per_node=1 --master_port 1114 parsing/run.py --cfg experiments/HowTo100M/1Tars-PTCPCFGs.yaml --test_mode --split test --checkpoint checkpoints/HowTo100M/1Tars-PTCPCFGs/seed-1119/epoch0000.pkl
python -m torch.distributed.launch --nproc_per_node=1 --master_port 1114 parsing/run.py --cfg experiments/HowTo100M/1Tars-PTCPCFGs.yaml --test_mode --split test --checkpoint checkpoints/HowTo100M/1Tars-PTCPCFGs/seed-1120/epoch0000.pkl
python -m torch.distributed.launch --nproc_per_node=1 --master_port 1114 parsing/run.py --cfg experiments/HowTo100M/1Tars-PTCPCFGs.yaml --test_mode --split test --checkpoint checkpoints/HowTo100M/1Tars-PTCPCFGs/seed-1121/epoch0000.pkl

echo '2Tars-PTCPCFGs'
python -m torch.distributed.launch --nproc_per_node=1 --master_port 1114 parsing/run.py --cfg experiments/HowTo100M/2Tars-PTCPCFGs.yaml --test_mode --split test --checkpoint checkpoints/HowTo100M/2Tars-PTCPCFGs/seed-1110/epoch0000.pkl
python -m torch.distributed.launch --nproc_per_node=1 --master_port 1114 parsing/run.py --cfg experiments/HowTo100M/2Tars-PTCPCFGs.yaml --test_mode --split test --checkpoint checkpoints/HowTo100M/2Tars-PTCPCFGs/seed-1117/epoch0000.pkl
python -m torch.distributed.launch --nproc_per_node=1 --master_port 1114 parsing/run.py --cfg experiments/HowTo100M/2Tars-PTCPCFGs.yaml --test_mode --split test --checkpoint checkpoints/HowTo100M/2Tars-PTCPCFGs/seed-1119/epoch0000.pkl
python -m torch.distributed.launch --nproc_per_node=1 --master_port 1114 parsing/run.py --cfg experiments/HowTo100M/2Tars-PTCPCFGs.yaml --test_mode --split test --checkpoint checkpoints/HowTo100M/2Tars-PTCPCFGs/seed-1120/epoch0000.pkl
python -m torch.distributed.launch --nproc_per_node=1 --master_port 1114 parsing/run.py --cfg experiments/HowTo100M/2Tars-PTCPCFGs.yaml --test_mode --split test --checkpoint checkpoints/HowTo100M/2Tars-PTCPCFGs/seed-1121/epoch0000.pkl

echo '4Tars-PTCPCFGs'
python -m torch.distributed.launch --nproc_per_node=1 --master_port 1114 parsing/run.py --cfg experiments/HowTo100M/4Tars-PTCPCFGs.yaml --test_mode --split test --checkpoint checkpoints/HowTo100M/4Tars-PTCPCFGs/seed-1110/epoch0000.pkl
python -m torch.distributed.launch --nproc_per_node=1 --master_port 1114 parsing/run.py --cfg experiments/HowTo100M/4Tars-PTCPCFGs.yaml --test_mode --split test --checkpoint checkpoints/HowTo100M/4Tars-PTCPCFGs/seed-1117/epoch0000.pkl
python -m torch.distributed.launch --nproc_per_node=1 --master_port 1114 parsing/run.py --cfg experiments/HowTo100M/4Tars-PTCPCFGs.yaml --test_mode --split test --checkpoint checkpoints/HowTo100M/4Tars-PTCPCFGs/seed-1119/epoch0000.pkl
python -m torch.distributed.launch --nproc_per_node=1 --master_port 1114 parsing/run.py --cfg experiments/HowTo100M/4Tars-PTCPCFGs.yaml --test_mode --split test --checkpoint checkpoints/HowTo100M/4Tars-PTCPCFGs/seed-1120/epoch0000.pkl
python -m torch.distributed.launch --nproc_per_node=1 --master_port 1114 parsing/run.py --cfg experiments/HowTo100M/4Tars-PTCPCFGs.yaml --test_mode --split test --checkpoint checkpoints/HowTo100M/4Tars-PTCPCFGs/seed-1121/epoch0000.pkl

echo '8Tars-PTCPCFGs'
python -m torch.distributed.launch --nproc_per_node=1 --master_port 1114 parsing/run.py --cfg experiments/HowTo100M/8Tars-PTCPCFGs.yaml --test_mode --split test --checkpoint checkpoints/HowTo100M/8Tars-PTCPCFGs/seed-1110/epoch0000.pkl
python -m torch.distributed.launch --nproc_per_node=1 --master_port 1114 parsing/run.py --cfg experiments/HowTo100M/8Tars-PTCPCFGs.yaml --test_mode --split test --checkpoint checkpoints/HowTo100M/8Tars-PTCPCFGs/seed-1117/epoch0000.pkl
python -m torch.distributed.launch --nproc_per_node=1 --master_port 1114 parsing/run.py --cfg experiments/HowTo100M/8Tars-PTCPCFGs.yaml --test_mode --split test --checkpoint checkpoints/HowTo100M/8Tars-PTCPCFGs/seed-1119/epoch0000.pkl
python -m torch.distributed.launch --nproc_per_node=1 --master_port 1114 parsing/run.py --cfg experiments/HowTo100M/8Tars-PTCPCFGs.yaml --test_mode --split test --checkpoint checkpoints/HowTo100M/8Tars-PTCPCFGs/seed-1120/epoch0000.pkl
python -m torch.distributed.launch --nproc_per_node=1 --master_port 1114 parsing/run.py --cfg experiments/HowTo100M/8Tars-PTCPCFGs.yaml --test_mode --split test --checkpoint checkpoints/HowTo100M/8Tars-PTCPCFGs/seed-1121/epoch0000.pkl
