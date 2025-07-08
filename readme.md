# Apply Deit on Smh_detection Project

## 之前的环境理论上就可以跑这个代码，如果之前未在环境中使用过wandb, 则在命令行中使用`wandb login`指令并根据提示粘贴密钥`333a3c5e229352e46223a230739cd4a6c0b175c6`，成功连接后，在代码运行时应该能看到wandb的相关提示

## 已在`/home/shared_data/`下创建一个咱俩都看的见的文件夹`/home/shared_data/salmonella_detection/smh_shared_output`来做为输出结果保存地址,请测试一下是否有访问权限，可执行以下命令：
```
cd /home/shared_data/salmonella_detection/smh_shared_output
cat test_from_checne.txt
```
## 然后每个实验尽量用单独的文件夹保存，不要把文件都放在这个下面，后面不好管理
## debug使用launch.json
## 使用这个命令运行代码，把里面关于名称的变量可以换一换
```
python main.py --model deit_tiny_patch16_224 --batch-size 64 --data-set Smh_custom --data-path /home/shared_data/salmonella_detection/OriginalData/AmericanData --output_dir /home/shared_data/salmonella_detection/smh_shared_output/exp_name -logger -exp_name first_try
```
