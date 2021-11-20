# MSR-GCN

## Overview




  &nbsp;&nbsp;&nbsp;  In this project, we integrated discrete cosine transformation with a multi-scale residual graph convolution network in this project to accurately predict future human movements from recorded histories. To offer intermediate oversight, losses are added to all scales. We predict 25 frames in the future using a brief recorded historical posture sequence of 10 frames as input. On the entire test dataset, we test and compare the suggested strategy with prior state-of-the-art methodologies. On two typical benchmark datasets, our methodology beats the state-of-the-art approaches

## Dependencies

* Pytorch 1.7.0+cu110
* Python 3.8.5
* Nvidia RTX 3090

## Get the data
[Human3.6m](http://vision.imar.ro/human3.6m/description.php) in exponential map can be downloaded from [here](http://www.cs.stanford.edu/people/ashesh/h3.6m.zip).

[CMU mocap](http://mocap.cs.cmu.edu/) was obtained from the [repo](https://github.com/chaneyddtt/Convolutional-Sequence-to-Sequence-Model-for-Human-Dynamics) of ConvSeq2Seq paper.

## About datasets

Human3.6M

+ A pose in h3.6m has 32 joints, from which we choose 22, and build the multi-scale by 22 -> 12 -> 7 -> 4 dividing manner.
+ We use S5 / S11 as test / valid dataset, and the rest as train dataset, testing is done on the 15 actions separately, on each we use all data instead of the randomly selected 8 samples.
+ Some joints of the origin 32 have the same position
+ The input / output length is 10 / 25

CMU Mocap dataset

+ A pose in cmu has 38 joints, from which we choose 25, and build the multi-scale by 25 -> 12 -> 7 -> 4 dividing manner.
+ CMU does not have valid dataset, testing is done on the 8 actions separately, on each we use all data instead of the random selected 8 samples.
+ Some joints of the origin 38 have the same position
+ The input / output length is 10 / 25

## Train

+ train on Human3.6M:

  `python main.py --exp_name=h36m --is_train=1 --output_n=20 --dct_n=50 --test_manner=all --dct=true`

+ train on CMU Mocap:

  `python main.py --exp_name=cmu --is_train=1 --output_n=20 --dct_n=50 --test_manner=all --dct=true`


## Evaluate and visualize results

+ evaluate on Human3.6M:

  `python main.py --exp_name=h36m --is_load=1 --model_path= <path to .pth file generated after training> --output_n=50 --dct_n=50 --test_manner=all`

+ evaluate on CMU Mocap: 
  
  `python main.py --exp_name=cmu --is_load=1 --model_path=<path to .pth file generated after training> --output_n=50 --dct_n=50 --test_manner=all`

## Results

H3.6M-20/50/50-all| 80                 | 160                | 320                | 400                | 560                | 1000               | 
|-----------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|
| walking         | 11.473099989370938 | 24.624715095731972 | 38.55695128151828  | 46.485453865409    | 54.09193766083292  | 63.60718147907753  |
| eating          | 8.241971325664728  | 15.820354610064904 | 31.103194199398384 | 40.26217926430223  | 52.27206544193631  | 77.3673984154661   |
| smoking         | 7.840467137789415  | 14.982100548108887 | 29.699002742012116 | 39.483276536410415 | 49.560906652284075 | 73.0815682303758   |
| discussion      | 12.076533687177381 | 28.169655963826145 | 58.637938025436746 | 71.1026042068472   | 89.77804232738842  | 115.86206428992429 |
| directions      | 7.0901809740982    | 18.83952788354761  | 45.25037403597105  | 54.07548905053886  | 69.72809388261439  | 101.42853340763372 |
| greeting        | 14.867320929462409 | 35.660891338034574 | 77.26229165444155  | 95.31917893782104  | 116.99108407532518 | 145.73854156614973 |
| phoning         | 10.628145417830908 | 20.140647569033057 | 40.12352180853908  | 51.59667678902612  | 69.75929481687015  | 104.32955535575844 |
| posing          | 13.623185960863827 | 28.198596324903303 | 66.86566589661822  | 84.87406825233613  | 116.75462255680925 | 172.54827358671136 |
| purchases       | 15.958232028167384 | 31.470628855929263 | 65.77912421210594  | 81.50556742044536  | 103.29616854055429 | 140.1607356300925  |
| sitting         | 11.299604707984333 | 21.990238058829693 | 46.07921315810751  | 59.52790379916207  | 77.22196270422052  | 119.27347416717377 |
| sittingdown     | 14.840941141499382 | 32.17434049907326  | 62.90062621354873  | 75.48502872091632  | 103.94367646393101 | 157.25899622486847 |
| takingphoto     | 11.12058013209418  | 22.186785879970294 | 46.53823168878216  | 57.23889529967435  | 77.71301640112748  | 122.86793247257235 |
| waiting         | 11.30685252623092  | 21.23182028531109  | 48.53389942331784  | 60.45024104186935  | 76.0864070695311   | 104.83289295996109 |
| walkingdog      | 20.884435964070388 | 41.06234620668756  | 80.32721643067664  | 91.3070495621277   | 112.33347638112562 | 147.36855750014266 |
| walkingtogether | 11.154736960911867 | 22.077251540678667 | 35.60193419253564  | 43.08125386155917  | 53.303694772974595 | 64.22133718583802  |

****

|CMU-10/25/35-all   | 80                 | 160                | 320                 | 400                | 560                 | 1000               |
|-------------------|--------------------|--------------------|---------------------|--------------------|---------------------|--------------------|
| basketball        | 10.242487498346153 | 18.63962115638782  | 36.04137525902938   | 45.855150906289217 | 60.629692129712631  | 86.935806856317623 |
| basketball_signal | 3.0052560005885107 | 6.3692770854203584 | 12.9509714775020516 | 16.97734159921188  | 27.1635307589277679 | 49.903279756016    |
| directing_traffic | 6.003408082892022  | 14.003408082892022 | 29.6349963023095    | 37.220475428520    | 60.488932513307664  | 115.1975532513753  |
| jumping           | 15.684301666213866 | 28.9468790585122   | 57.395946892827031  | 69.062247732636693 | 92.85818992033665   | 126.4195408609157  |
| running           | 16.557456389834150 | 21.851476378877315 | 30.229165809096323  | 33.038315146077360 | 35.6914941522888    | 41.6008927171899   |
| soccer            | 11.793302776898417 | 19.545640223751613 | 35.4551267498754    | 46.931107470694894 | 65.89817177263369   | 100.54471112731630 |
| walking           | 6.38907388294679   | 10.60768053893843  | 16.80164150194786   | 20.67280076960615  | 26.12820587608290   | 36.286492756191    |
| washwindow        | 6.27944197025156   | 11.55456460951392  | 24.94162868592245   | 29.64071033285404  | 46.02909824688188   | 70.51355700475477  |
