#!/bin/bash

RESULT_DIR="./result"

echo "开始运行实验..."

cd "$(dirname "$0")"

mkdir -p "$RESULT_DIR"

echo "开始运行 Bert4Rec.py..."
for i in {1..3}
do
    echo "运行 Bert4Rec.py 第 $i 次..."
    log_file="${RESULT_DIR}/Bert4Rec_${i}.log"
    python Bert4Rec.py --log_file "$log_file"
    echo "Bert4Rec.py 第 $i 次运行完成，日志保存到: $log_file"
    echo "----------------------------------------"
done

# echo "开始运行 Bert4Rec_RUIE.py..."
# for i in {1..3}
# do
#     echo "运行 Bert4Rec_RUIE.py 第 $i 次..."
#     log_file="${RESULT_DIR}/Bert4Rec_RUIE_${i}.log"
#     python Bert4Rec_RUIE.py --log_file "$log_file"
#     echo "Bert4Rec_RUIE.py 第 $i 次运行完成，日志保存到: $log_file"
#     echo "----------------------------------------"
# done

# echo "开始运行 HiNet.py..."
# for i in {1..5}
# do
#     echo "运行 HiNet.py 第 $i 次..."
#     log_file="${RESULT_DIR}/HiNet_${i}.log"
#     python HiNet.py --log_file "$log_file"
#     echo "HiNet.py 第 $i 次运行完成，日志保存到: $log_file"
#     echo "----------------------------------------"
# done

# echo "开始运行 HiNet_RUIE.py..."
# for i in {1..3}
# do
#     echo "运行 HiNet_RUIE.py 第 $i 次..."
#     log_file="${RESULT_DIR}/HiNet_RUIE_${i}.log"
#     python HiNet_RUIE.py --log_file "$log_file"
#     echo "HiNet_RUIE.py 第 $i 次运行完成，日志保存到: $log_file"
#     echo "----------------------------------------"
# done

# echo "开始运行 PEPNet.py..."
# for i in {1..5}
# do
#     echo "运行 PEPNet.py 第 $i 次..."
#     log_file="${RESULT_DIR}/PEPNet_${i}.log"
#     python PEPNet.py --log_file "$log_file"
#     echo "PEPNet.py 第 $i 次运行完成，日志保存到: $log_file"
#     echo "----------------------------------------"
# done

# echo "开始运行 PEPNet_RUIE.py..."
# for i in {1..3}
# do
#     echo "运行 PEPNet_RUIE.py 第 $i 次..."
#     log_file="${RESULT_DIR}/PEPNet_RUIE_${i}.log"
#     python PEPNet_RUIE.py --log_file "$log_file"
#     echo "PEPNet_RUIE.py 第 $i 次运行完成，日志保存到: $log_file"
#     echo "----------------------------------------"
# done

# echo "开始运行 STAR.py..."
# for i in {1..5}
# do
#     echo "运行 STAR.py 第 $i 次..."
#     log_file="${RESULT_DIR}/STAR_${i}.log"
#     python STAR.py --log_file "$log_file"
#     echo "STAR.py 第 $i 次运行完成，日志保存到: $log_file"
#     echo "----------------------------------------"
# done

# echo "开始运行 STAR_RUIE.py..."
# for i in {1..3}
# do
#     echo "运行 STAR_RUIE.py 第 $i 次..."
#     log_file="${RESULT_DIR}/STAR_RUIE_${i}.log"
#     python STAR_RUIE.py --log_file "$log_file"
#     echo "STAR_RUIE.py 第 $i 次运行完成，日志保存到: $log_file"
#     echo "----------------------------------------"
# done

echo "========================================"
echo "所有实验运行完成！"
