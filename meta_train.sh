python3 train.py --batch_size=16 --epochs=10 --learner_epochs=8 --num_classes=10 \
--steps_per_epoch_limit=200 --num_shot=5 --num_shot_eval=15 --use_bias=1 \
--image_x=28 --image_y=28 --filters=32 --padding=SAME --kernel_size=3 \
--num_blocks=2 --maxpool_size=2 --meta_hidden_size=20 --meta_input_size=4 \
--b_init_0=4 --b_init_1=5
