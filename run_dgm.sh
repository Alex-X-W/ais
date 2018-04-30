# primary comparison between VAE and GAN
echo '**************** primary cmoparison ***************' >> dgm_log
echo '**************** primary cmoparison ***************'
echo 'vae10...' >> dgm_log
echo 'vae10 on train'
python dgm_tester.py --exps train --model vae10 >> dgm_log
# echo 'vae10 on test'
# python dgm_tester.py --exps test --model vae10 >> dgm_log

echo 'gan10...' >> dgm_log
echo 'gan10 on train'
python dgm_tester.py --exps train --model gan10 >> dgm_log
# echo 'gan10 on test'
# python dgm_tester.py --exps test --model gan10 >> dgm_log

echo 'vae50...' >> dgm_log
echo 'vae50 on train'
python dgm_tester.py --exps train --model vae50 --hdim 50 >> dgm_log
# echo 'vae50 on test'
# python dgm_tester.py --exps test --model vae50 --hdim 50 >> dgm_log

echo 'gan50...' >> dgm_log
echo 'gan50 on train'
python dgm_tester.py --exps train --model gan50 --hdim 50 >> dgm_log
echo 'gan50 on test'
python dgm_tester.py --exps test --model gan50 --hdim 50 >> dgm_log


# evaluate GAN with varying observation model variance
# echo '**************** GAN varing obs var ***************' >> dgm_log
# echo '**************** GAN varing obs var ***************'
# echo 'gan10, sigma=0.001' >> dgm_log
# echo 'gan10, sigma=0.001'
# python dgm_tester.py --exps test --model gan10 --sigma 0.001 >> dgm_log
# echo 'gan10, sigma=0.005' >> dgm_log
# echo 'gan10, sigma=0.005'
# python dgm_tester.py --exps test --model gan10 --sigma 0.005 >> dgm_log
# echo 'gan10, sigma=0.01' >> dgm_log
# echo 'gan10, sigma=0.01'
# python dgm_tester.py --exps test --model gan10 --sigma 0.01 >> dgm_log
# echo 'gan10, sigma=0.05' >> dgm_log
# echo 'gan10, sigma=0.05'
# python dgm_tester.py --exps test --model gan10 --sigma 0.05 >> dgm_log
# echo 'gan10, sigma=0.1' >> dgm_log
# echo 'gan10, sigma=0.1'
# python dgm_tester.py --exps test --model gan10 --sigma 0.1 >> dgm_log
# echo 'gan10, sigma=0.5' >> dgm_log
# echo 'gan10, sigma=0.5'
# python dgm_tester.py --exps test --model gan10 --sigma 0.5 >> dgm_log