"""
Unit tests for the structured metamodel component.
"""
import unittest

import numpy as np
from numpy.testing import assert_almost_equal

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal, assert_warning, assert_check_partials

scipy_gte_019 = True
try:
    from scipy.interpolate._bsplines import make_interp_spline
except ImportError:
    scipy_gte_019 = False


x = np.array([-0.97727788, -0.15135721, -0.10321885,  0.40015721,  0.4105985,
               0.95008842,  0.97873798,  1.76405235,  1.86755799,  2.2408932 ])

y = np.array([ 0.12167502,  0.14404357,  0.44386323,  0.76103773,  1.45427351])

z = np.array([-2.55298982, -1.45436567, -0.85409574, -0.74216502, -0.20515826, 0.04575852,
               0.3130677,   0.33367433,  0.6536186,   0.8644362,  1.49407907,  2.26975462])

f = np.array([
    [[-0.18718385,  1.53277921,  1.46935877,  0.15494743,  0.37816252, -0.88778575,
      -1.98079647, -0.34791215,  0.15634897,  1.23029068,  1.20237985, -0.38732682],
     [-0.30230275, -1.04855297, -1.42001794, -1.70627019,  1.9507754,  -0.50965218,
      -0.4380743,  -1.25279536,  0.77749036, -1.61389785, -0.21274028, -0.89546656],
     [ 0.3869025,  -0.51080514, -1.18063218, -0.02818223,  0.42833187,  0.06651722,
       0.3024719,  -0.63432209, -0.36274117, -0.67246045, -0.35955316, -0.81314628],
     [-1.7262826,   0.17742614, -0.40178094, -1.63019835,  0.46278226, -0.90729836,
       0.0519454,   0.72909056,  0.12898291,  1.13940068, -1.23482582,  0.40234164],
     [-0.68481009, -0.87079715, -0.57884966, -0.31155253,  0.05616534, -1.16514984,
       0.90082649,  0.46566244, -1.53624369,  1.48825219,  1.89588918,  1.17877957]],

    [[-0.17992484, -1.07075262,  1.05445173, -0.40317695,  1.22244507,  0.20827498,
       0.97663904,  0.3563664,   0.70657317,  0.01050002,  1.78587049,  0.12691209],
     [ 0.40198936,  1.8831507,  -1.34775906, -1.270485,    0.96939671, -1.17312341,
       1.94362119, -0.41361898, -0.74745481,  1.92294203,  1.48051479,  1.86755896],
     [ 0.90604466, -0.86122569,  1.91006495, -0.26800337,  0.8024564,   0.94725197,
      -0.15501009,  0.61407937,  0.92220667,  0.37642553, -1.09940079,  0.29823817],
     [ 1.3263859,  -0.69456786, -0.14963454, -0.43515355,  1.84926373,  0.67229476,
       0.40746184, -0.76991607,  0.53924919, -0.67433266,  0.03183056, -0.63584608],
     [ 0.67643329,  0.57659082, -0.20829876,  0.39600671, -1.09306151, -1.49125759,
       0.4393917,   0.1666735,   0.63503144,  2.38314477,  0.94447949, -0.91282223]],

    [[ 1.11701629, -1.31590741, -0.4615846,  -0.06824161,  1.71334272, -0.74475482,
      -0.82643854, -0.09845252, -0.66347829,  1.12663592, -1.07993151, -1.14746865],
     [-0.43782004, -0.49803245,  1.92953205,  0.94942081,  0.08755124, -1.22543552,
       0.84436298, -1.00021535, -1.5447711,   1.18802979,  0.31694261,  0.92085882],
     [ 0.31872765,  0.85683061, -0.65102559, -1.03424284,  0.68159452, -0.80340966,
      -0.68954978, -0.4555325,   0.01747916, -0.35399391, -1.37495129, -0.6436184 ],
     [-2.22340315,  0.62523145, -1.60205766, -1.10438334,  0.05216508, -0.739563,
       1.5430146,  -1.29285691,  0.26705087, -0.03928282, -1.1680935,   0.52327666],
     [-0.17154633,  0.77179055,  0.82350415,  2.16323595,  1.33652795, -0.36918184,
      -0.23937918,  1.0996596,   0.65526373,  0.64013153, -1.61695604, -0.02432612]],

    [[-0.73803091,  0.2799246,  -0.09815039,  0.91017891,  0.31721822,  0.78632796,
      -0.4664191,  -0.94444626, -0.41004969, -0.01702041,  0.37915174,  2.25930895],
     [-0.04225715, -0.955945 ,  -0.34598178, -0.46359597,  0.48148147, -1.54079701,
       0.06326199,  0.15650654,  0.23218104, -0.59731607, -0.23792173, -1.42406091],
     [-0.49331988, -0.54286148,  0.41605005, -1.15618243,  0.7811981,   1.49448454,
      -2.06998503,  0.42625873,  0.67690804, -0.63743703, -0.39727181, -0.13288058],
     [-0.29779088, -0.30901297, -1.67600381,  1.15233156,  1.07961859, -0.81336426,
      -1.46642433,  0.52106488, -0.57578797,  0.14195316, -0.31932842,  0.69153875],
     [ 0.69474914, -0.72559738, -1.38336396, -1.5829384,   0.61037938, -1.18885926,
      -0.50681635, -0.59631404, -0.0525673,  -1.93627981,  0.1887786,   0.52389102]],

    [[ 0.08842209, -0.31088617,  0.09740017,  0.39904635, -2.77259276,  1.95591231,
       0.39009332, -0.65240858, -0.39095338,  0.49374178, -0.11610394, -2.03068447],
     [ 2.06449286, -0.11054066,  1.02017271, -0.69204985,  1.53637705,  0.28634369,
       0.60884383, -1.04525337,  1.21114529,  0.68981816,  1.30184623, -0.62808756],
     [-0.48102712,  2.3039167,  -1.06001582, -0.1359497,   1.13689136,  0.09772497,
       0.58295368, -0.39944903,  0.37005589, -1.30652685,  1.65813068, -0.11816405],
     [-0.6801782,   0.66638308, -0.46071979, -1.33425847, -1.34671751,  0.69377315,
      -0.15957344, -0.13370156,  1.07774381, -1.12682581, -0.73067775, -0.38487981],
     [ 0.09435159, -0.04217145, -0.28688719, -0.0616264,  -0.10730528, -0.71960439,
      -0.81299299,  0.27451636, -0.89091508, -1.15735526, -0.31229225, -0.15766702]],

    [[ 2.2567235,  -0.70470028,  0.94326072,  0.74718833, -1.18894496,  0.77325298,
      -1.18388064, -2.65917224,  0.60631952, -1.75589058,  0.45093446, -0.6840109 ],
     [ 1.6595508,   1.0685094,  -0.4533858,  -0.68783761, -1.2140774,  -0.44092263,
      -0.2803555,  -0.36469354,  0.15670386,  0.5785215,   0.34965446, -0.76414392],
     [-1.43779147,  1.36453185, -0.68944918, -0.6522936,  -0.52118931, -1.84306955,
      -0.477974 ,  -0.47965581,  0.6203583,   0.69845715,  0.00377089,  0.93184837],
     [ 0.33996498, -0.01568211,  0.16092817, -0.19065349, -0.39484951, -0.26773354,
      -1.12801133,  0.28044171, -0.99312361,  0.84163126, -0.24945858,  0.04949498],
     [ 0.49383678,  0.64331447, -1.57062341, -0.20690368,  0.88017891, -1.69810582,
       0.38728048, -2.25556423, -1.02250684,  0.03863055, -1.6567151,  -0.98551074]],

    [[-1.47183501,  1.64813493,  0.16422776,  0.56729028, -0.2226751,  -0.35343175,
      -1.61647419, -0.29183736, -0.76149221,  0.85792392,  1.14110187,  1.46657872],
     [ 0.85255194, -0.59865394, -1.11589699,  0.76666318,  0.35629282, -1.76853845,
       0.35548179,  0.81451982,  0.05892559, -0.18505367, -0.80764849, -1.4465347 ],
     [ 0.80029795, -0.30911444, -0.23346666,  1.73272119,  0.68450111,  0.370825 ,
       0.14206181,  1.51999486,  1.71958931,  0.92950511,  0.58222459, -2.09460307],
     [ 0.12372191, -0.13010695,  0.09395323,  0.94304609, -2.73967717, -0.56931205,
       0.26990435, -0.46684555, -1.41690611,  0.86896349,  0.27687191, -0.97110457],
     [ 0.3148172,   0.82158571,  0.00529265,  0.8005648,   0.07826018, -0.39522898,
      -1.15942052, -0.08593077,  0.19429294,  0.87583276, -0.11510747,  0.45741561]],

    [[-0.96461201, -0.78262916, -0.1103893,  -1.05462846,  0.82024784,  0.46313033,
       0.27909576,  0.33890413,  2.02104356, -0.46886419, -2.20144129,  0.1993002 ],
     [-0.05060354, -0.51751904, -0.97882986, -0.43918952,  0.18133843, -0.5028167,
       2.41245368, -0.96050438, -0.79311736, -2.28862004,  0.25148442, -2.01640663],
     [-0.53945463, -0.27567053, -0.70972797,  1.73887268,  0.99439439,  1.31913688,
      -0.88241882,  1.12859406,  0.49600095,  0.77140595,  1.02943883, -0.90876325],
     [-0.42431762,  0.86259601, -2.65561909,  1.51332808,  0.55313206, -0.04570396,
       0.22050766, -1.02993528, -0.34994336,  1.10028434,  1.29802197,  2.69622405],
     [-0.07392467, -0.65855297, -0.51423397, -1.01804188, -0.07785476,  0.38273243,
      -0.03424228,  1.09634685, -0.2342158,  -0.34745065, -0.58126848, -1.63263453]],

    [[-1.56776772, -1.17915793,  1.30142807,  0.89526027,  1.37496407, -1.33221165,
      -1.96862469, -0.66005632,  0.17581895,  0.49869027,  1.04797216,  0.28427967],
     [ 1.74266878, -0.22260568, -0.91307922, -1.68121822, -0.88897136,  0.24211796,
      -0.88872026,  0.93674246,  1.41232771, -2.36958691,  0.8640523,  -2.23960406],
     [ 0.40149906,  1.22487056,  0.06485611, -1.27968917, -0.5854312,  -0.26164545,
      -0.18224478, -0.20289684, -0.10988278,  0.21348005, -1.20857365, -0.24201983],
     [ 1.51826117, -0.38464542, -0.44383609,  1.0781973,  -2.55918467,  1.1813786,
      -0.63190376,  0.16392857,  0.09632136,  0.94246812, -0.26759475, -0.67802578],
     [ 1.29784579, -2.36417382,  0.02033418, -1.34792542, -0.76157339,  2.01125668,
      -0.04459543,  0.1950697,  -1.78156286, -0.72904466,  0.1965574,   0.35475769]],

    [[ 0.61688655,  0.0086279,   0.52700421,  0.45378191, -1.82974041,  0.03700572,
       0.76790241,  0.58987982, -0.36385881, -0.80562651, -1.11831192, -0.13105401],
     [ 1.13307988, -1.9518041,  -0.65989173, -1.13980246,  0.78495752, -0.55430963,
      -0.47063766, -0.21694957,  0.44539325, -0.392389,   -3.04614305,  0.54331189],
     [ 0.43904296, -0.21954103, -1.08403662,  0.35178011,  0.37923553, -0.47003288,
      -0.21673147, -0.9301565,  -0.17858909, -1.55042935,  0.41731882, -0.94436849],
     [ 0.23810315, -1.40596292, -0.59005765, -0.11048941, -1.66069981,  0.11514787,
      -0.37914756, -1.7423562,  -1.30324275,  0.60512008,  0.89555599, -0.13190864],
     [ 0.40476181,  0.22384356,  0.32962298,  1.28598401, -1.5069984,   0.67646073,
      -0.38200896, -0.22425893, -0.30224973, -0.37514712, -1.22619619,  0.1833392 ]
]])

g = np.array([
    [[ 1.67094303e+00,  -5.61330204e-02,  -1.38504274e-03,
      -6.87299037e-01,  -1.17474546e-01,   4.66166426e-01,
      -3.70242441e-01,  -4.53804041e-01,   4.03264540e-01,
      -9.18004770e-01,   2.52496627e-01,   8.20321797e-01],
     [ 1.35994854e+00,  -9.03820073e-02,   1.36759724e+00,
       1.03440989e+00,  -9.96212640e-01,  -1.21793851e+00,
      -3.04963638e-01,   1.02893549e+00,  -7.22870076e-02,
      -6.00657558e-01,   1.55224318e+00,   2.86904488e-01],
     [-2.32059428e+00,   3.17160626e-01,   5.20040615e-01,
       2.25608654e-01,   4.49712100e-01,  -6.72756089e-02,
      -1.31839587e+00,  -3.70704003e-01,  -9.45615796e-01,
      -9.32740911e-01,  -1.26306835e+00,   4.52489093e-01],
     [ 9.78961454e-02,  -4.48165363e-01,  -6.49337928e-01,
      -2.34231050e-02,   1.07919473e+00,  -2.00421572e+00,
       3.76876521e-01,  -5.45711974e-01,  -1.88458584e+00,
      -1.94570308e+00,  -9.12783494e-01,   2.19509556e-01],
     [ 3.93062934e-01,  -9.38981573e-01,   1.01702099e+00,
       1.42298350e+00,   3.96086585e-01,  -5.91402668e-01,
       1.12441918e+00,   7.55395696e-01,   8.67407411e-01,
      -6.56463675e-01,  -2.83455451e+00,   2.11679102e+00]],

    [[-1.61087840e+00,  -3.57680719e-02,   2.38074535e+00,
       3.30576756e-01,   9.49246474e-01,  -1.50239657e+00,
      -1.77766695e+00,  -5.32702792e-01,   1.09074973e+00,
      -3.46249448e-01,  -7.94636321e-01,   1.97967290e-01],
     [ 1.08193522e+00,  -1.44494020e+00,  -1.21054299e+00,
      -7.88669255e-01,   1.09463837e+00,   2.34821526e-01,
       2.13215341e+00,   9.36445726e-01,  -3.50951769e-02,
       1.26507784e+00,   2.11497013e-01,  -7.04921353e-01],
     [ 6.79974844e-01,  -6.96326654e-01,  -2.90397101e-01,
       1.32778270e+00,  -1.01281486e-01,  -8.03141387e-01,
      -4.64337691e-01,   1.02179059e+00,  -5.52540673e-01,
      -3.86870847e-01,  -5.10292740e-01,   1.83925494e-01],
     [-3.85489760e-01,  -1.60183605e+00,  -8.87180942e-01,
      -9.32789042e-01,   1.24331938e+00,   8.12674042e-01,
       5.87259379e-01,  -5.05358317e-01,  -8.15791542e-01,
      -5.07517602e-01,  -1.05188010e+00,   2.49720039e+00],
     [-2.24532165e+00,   5.64008535e-01,  -1.28455230e+00,
      -1.04343491e-01,  -9.88001942e-01,  -1.17762896e+00,
      -1.14019630e+00,   1.75498615e+00,  -1.32988422e-01,
      -7.65702194e-01,   5.55786964e-01,   1.03493146e-02]],

    [[ 7.20033759e-01,  -1.82425666e+00,   3.03603904e-01,
       7.72694837e-01,  -1.66159829e+00,   4.48195284e-01,
       1.69618157e+00,  -1.48577034e-02,   8.21405937e-01,
       6.70570450e-01,  -7.07505698e-01,   3.97667346e-02],
     [-1.56699471e+00,  -4.51303037e-01,   2.65687975e-01,
       7.23100494e-01,   2.46121252e-02,   7.19983730e-01,
      -1.10290621e+00,  -1.01697275e-01,   1.92793845e-02,
       1.84959125e+00,  -2.14166656e-01,  -4.99016638e-01],
     [ 2.13512238e-02,  -9.19113445e-01,   1.92753849e-01,
      -3.65055217e-01,  -1.79132755e+00,  -5.85865511e-02,
      -3.17543094e-01,  -1.63242330e+00,  -6.71341546e-02,
       1.48935596e+00,   5.21303748e-01,   6.11927193e-01],
     [-1.34149673e+00,   4.76898369e-01,   1.48449581e-01,
       5.29045238e-01,   4.22628622e-01,  -1.35978073e+00,
      -4.14008116e-02,  -7.57870860e-01,  -5.00840943e-02,
      -8.97400927e-01,   1.31247037e+00,  -8.58972388e-01],
     [-8.98942156e-01,   7.45864065e-02,  -1.07709907e+00,
      -4.24663302e-01,  -8.29964598e-01,   1.41117206e+00,
       7.85803827e-01,  -5.74695185e-02,  -3.91217052e-01,
       9.40917615e-01,   4.05204080e-01,   4.98052405e-01]],

    [[-2.61922373e-02,  -1.68823003e+00,  -1.12465983e-01,
      -5.32489919e-01,   6.45055273e-01,   1.01184243e+00,
      -6.57951045e-01,   4.68385234e-01,   1.73587900e+00,
      -6.67712721e-01,   1.68192174e+00,  -8.52585847e-01],
     [ 2.29597556e-02,  -1.11456118e-02,   1.14988999e-02,
      -8.37678042e-01,  -5.91183104e-01,  -6.67720286e-01,
       3.26962595e-01,   3.30035115e-01,   2.22594433e+00,
       1.37098901e+00,  -5.09843242e-01,   3.24869616e-01],
     [ 9.97117981e-01,   3.06018243e-02,  -6.96415784e-02,
       5.15749428e-02,   8.67276629e-01,  -8.48320523e-01,
      -3.25669469e-01,   4.70433145e-01,   3.11447072e-01,
       2.39582760e-01,  -3.69801166e-01,   9.72535789e-01],
     [ 2.13386825e+00,   4.06415494e-01,  -1.93176702e-01,
       7.55740289e-01,  -5.39132637e-01,  -7.49690345e-01,
       3.28087476e-02,  -2.58279663e+00,  -1.15395036e+00,
      -3.47961856e-01,  -1.35338886e+00,  -1.03264310e+00],
     [-4.36748337e-01,  -1.64296529e+00,  -4.06071796e-01,
      -5.35270165e-01,   2.54052084e-02,   1.15418403e+00,
       1.72504416e-01,   2.10620213e-02,   9.94544570e-02,
       2.27392775e-01,  -1.01673865e+00,  -1.14775325e-01]],

    [[ 3.08751242e-01,  -1.37075998e+00,   8.65652923e-01,
       1.08137603e+00,  -6.31375988e-01,  -2.41337791e-01,
      -8.78190343e-01,   6.99380484e-01,  -1.06122229e+00,
      -2.22477010e-01,  -8.58919908e-01,   5.09542770e-02],
     [-1.79422927e+00,   1.32646164e+00,  -9.64606424e-01,
       5.98946831e-02,  -2.12523045e-01,  -7.62114512e-01,
      -8.87780137e-01,   9.36398544e-01,  -5.25640593e-01,
       2.71170185e-01,  -8.01496885e-01,  -6.47181432e-01],
     [ 4.72247150e-01,   9.30408496e-01,  -1.75316402e-01,
      -1.42191987e+00,   1.99795608e+00,  -8.56549308e-01,
      -1.54158740e+00,   2.59442459e+00,  -4.04032294e-01,
      -1.46173269e+00,  -6.83439767e-01,   3.67544896e-01],
     [ 1.90311558e-01,  -8.51729197e-01,   1.82272360e+00,
      -5.21579678e-01,  -1.18468659e+00,   9.60693398e-01,
       1.32906285e+00,  -8.17493098e-01,  -1.40134729e+00,
       1.03043827e+00,  -2.04732361e+00,  -1.22662166e+00],
     [ 9.67446150e-01,  -5.53525480e-02,  -2.63937349e-01,
       3.52816606e-01,  -1.52774424e-01,  -1.29868672e+00,
       1.27607535e+00,   1.32501405e+00,   2.05332564e-01,
       4.51340154e-02,   2.33962481e+00,  -2.76432845e-01]],

    [[-2.59576982e-01,   3.64481249e-01,   1.47132196e+00,
       1.59277075e+00,  -2.58572632e-01,   3.08331246e-01,
      -1.37808347e+00,  -3.11976108e-01,  -8.40290395e-01,
      -1.00683175e+00,   1.68157672e+00,  -7.92286662e-01],
     [-5.31605908e-01,   3.65848788e-01,   1.29782527e+00,
       4.81115126e-01,   2.75935511e+00,  -7.46679783e-02,
       2.58716440e-01,   2.75600674e-01,   1.43504939e+00,
       5.07238951e-01,  -1.16229700e-01,  -9.47488595e-01],
     [ 2.44443456e-01,   1.40134483e+00,  -4.10381794e-01,
       5.28943618e-01,   2.46147789e-01,   8.63519658e-01,
      -8.04753741e-01,   2.34664703e+00,  -1.27916111e+00,
      -3.65551090e-01,   9.38092541e-01,   2.96733172e-01],
     [ 8.29986159e-01,  -4.96102334e-01,  -7.48049827e-02,
       1.22319836e-02,   1.56925961e+00,   6.90429024e-01,
       7.96672108e-01,  -6.57926093e-01,   9.68882639e-01,
       2.25581664e-01,   1.38914532e+00,   2.01406015e+00],
     [-3.06765776e-01,  -4.06303130e-01,  -8.64044991e-01,
      -1.43579512e-01,  -3.82025449e-01,   3.59504400e-01,
      -1.44566817e-01,  -3.61599281e-01,   1.06458514e+00,
      -9.37880231e-01,   4.33107953e-01,  -4.05941727e-01]],

    [[ 7.24368505e-01,   1.38526155e+00,  -3.03098253e-01,
       4.41032907e-01,   1.78792866e-01,  -7.99422400e-01,
       2.40787510e-01,   2.89120505e-01,   4.12870820e-01,
      -1.98398897e-01,   9.41923003e-02,  -1.14761094e+00],
     [-3.58114075e-01,   5.55962680e-01,   8.92473887e-01,
      -4.22314824e-01,   1.04714029e-01,   2.28053325e-01,
       2.01479947e-01,   5.40773585e-01,  -1.81807763e+00,
      -4.93240701e-02,   2.39033601e-01,  -1.00033035e+00],
     [ 1.67398571e+00,   1.61559267e-01,   1.56340475e+00,
      -7.90523022e-01,  -9.07300122e-01,   2.24252221e-01,
      -1.67868836e+00,   2.14965591e-01,   9.72192320e-02,
       1.01566528e+00,   7.01041341e-01,  -4.17477350e-01],
     [-1.09749665e+00,   1.71230522e+00,  -7.92115021e-01,
      -1.04552456e+00,  -1.08485606e+00,   1.11730532e+00,
      -5.18900204e-01,  -7.53704466e-01,   1.37689826e-01,
      -2.06944711e-01,  -6.78095461e-01,   7.53991467e-01],
     [ 1.06531549e+00,   9.85317509e-01,   7.66919670e-01,
       4.02625531e-01,  -1.77588800e+00,   1.66925081e+00,
       3.01989210e-01,   6.08156428e-01,   1.11496232e+00,
       1.43335250e+00,   4.18398011e-01,   4.35546159e-01]],

    [[-5.99224277e-01,   3.30897511e-02,  -8.54161261e-01,
      -7.19940532e-01,  -8.93574402e-01,  -1.56023891e-01,
       1.04909319e+00,   3.17097477e+00,   1.89499638e-01,
      -1.34841309e+00,   1.26498333e+00,  -3.00783876e-01],
     [-6.60608594e-01,   2.09849478e-01,  -1.24062460e+00,
       2.22463164e-01,  -8.83755232e-02,   9.83779068e-02,
       3.81416254e-01,   6.74922572e-02,   1.63380841e-02,
       2.84314519e-01,   4.15400626e-01,  -1.03148246e+00],
     [-1.42999126e+00,  -6.16380522e-02,  -1.43273549e+00,
       8.75314709e-02,   9.38746876e-01,   6.07111672e-01,
      -1.04817041e+00,  -8.60262452e-01,   3.28301295e-01,
      -4.01297805e-01,  -3.16655295e-01,   5.96906481e-01],
     [-9.87286693e-01,  -4.01234710e-01,  -8.00082476e-01,
      -1.04312950e+00,  -8.57078189e-01,   6.77462169e-01,
       5.18203895e-02,  -8.79160629e-01,  -2.31101608e-01,
      -1.63880731e+00,  -7.33312808e-01,   2.14957453e+00],
     [-9.02438497e-02,   7.31658927e-01,  -6.54883751e-02,
       3.48169235e-01,   6.63258090e-01,  -1.10461660e+00,
      -3.09362573e-02,   1.57886519e+00,  -7.95500550e-01,
      -5.66439854e-01,  -3.07691277e-01,   2.69024073e-01]],

    [[ 5.24917864e-01,   1.26741165e+00,   4.99498233e-01,
      -6.20531258e-02,   1.25916713e+00,   7.04111022e-01,
      -1.49567952e+00,   2.52636824e+00,   1.76992139e+00,
      -1.68214223e-01,   3.77910102e-01,   1.32435875e+00],
     [-1.72200793e-01,   7.30351790e-01,   1.10457847e+00,
      -1.01482591e+00,  -6.02331854e-01,   9.21408398e-01,
       4.60814477e-01,   9.23796560e-01,  -1.32568015e-01,
      -2.89005211e-01,  -1.99863948e+00,  -1.14600043e+00],
     [ 4.70660947e-02,   8.24557220e-01,   5.31178367e-01,
      -1.28241974e-01,  -2.71771566e-01,   2.17179633e-01,
       7.82111811e-02,   1.40454551e+00,   1.46440770e-01,
      -1.48124596e+00,  -1.27255814e+00,   1.51875934e+00],
     [-1.17116046e+00,   7.64497453e-01,  -2.68372735e-01,
      -1.69758294e-01,  -1.34132783e-01,   1.22138496e+00,
      -1.92841829e-01,  -3.33192828e-02,  -1.53080350e+00,
       2.06690512e-01,   5.31042507e-01,   2.39145581e-01],
     [ 1.39789626e+00,   5.51713548e-02,   2.98977456e-01,
       1.64850401e+00,  -1.55001419e+00,  -4.55825348e-01,
       1.42615875e+00,   9.36129148e-01,   6.78380099e-01,
       8.32650739e-01,   3.27066209e-01,   1.63159743e+00]],

    [[ 3.77759170e-01,   2.39867106e-01,   1.58958674e-01,
       1.92863956e-01,  -1.15701728e+00,   7.70673054e-01,
      -1.30439734e-01,   1.82191510e+00,  -7.56504706e-02,
       4.20918284e-01,   2.46602186e-01,  -6.25557035e-01],
     [ 9.92136829e-01,   1.90506364e+00,  -1.47772197e-02,
      -3.00478786e-01,  -3.55028731e-01,  -1.89236189e+00,
      -1.77813144e-01,   2.50998116e-01,   1.05475793e+00,
       9.60047741e-01,  -4.16499082e-01,  -2.76822995e-01],
     [ 1.12390531e+00,  -1.73463897e-01,  -5.10029540e-01,
       1.39251845e+00,   1.03758567e+00,   1.87917918e-02,
      -5.93777448e-01,  -2.01188032e+00,   5.89703606e-01,
      -8.96369723e-01,  -1.96273201e+00,   1.58482053e+00],
     [ 6.47967791e-01,  -1.13900819e+00,  -1.21440138e+00,
       8.70961782e-01,  -8.77970617e-01,   1.29614987e+00,
       6.16459313e-01,   5.36596521e-01,   4.04695456e-01,
       1.91450872e-01,   8.80511199e-01,  -4.54080363e-01],
     [ 8.59519734e-02,   7.51946588e-01,   5.62989719e-01,
      -1.19498681e+00,  -5.00409667e-01,   2.52803505e-01,
      -4.08014709e-01,   1.77465856e+00,  -3.93153195e-01,
      -1.62218448e-01,   7.69430178e-01,   3.30532743e-01]]
])


class SampleMap(object):
    param_data = []
    np.random.seed(0)
    param_data.append({'name': 'x',
                       'units': None,
                       'default': 0,
                       'values': x})
    param_data.append({'name': 'y',
                       'units': None,
                       'default': 0,
                       'values': y})
    param_data.append({'name': 'z',
                       'units': None,
                       'default': 0,
                       'values': z})

    output_data = []
    output_data.append({'name': 'f',
                       'units': None,
                       'default': 0,
                       'values': f})

    output_data.append({'name': 'g',
                       'units': None,
                       'default': 0,
                       'values': g})


@unittest.skipIf(not scipy_gte_019, "only run if scipy>=0.19.")
class TestMetaModelStructuredScipy(unittest.TestCase):
    """
    Tests the regular grid map component. specifically the analytic derivatives
    vs. finite difference estimates.
    """

    def setUp(self):

        model = om.Group()
        ivc = om.IndepVarComp()

        mapdata = SampleMap()

        params = mapdata.param_data
        x, y, z = params
        outs = mapdata.output_data
        z = outs[0]
        ivc.add_output('x', x['default'], units=x['units'])
        ivc.add_output('y', y['default'], units=y['units'])
        ivc.add_output('z', z['default'], units=z['units'])

        model.add_subsystem('des_vars', ivc, promotes=["*"])

        comp = om.MetaModelStructuredComp(method='scipy_slinear', extrapolate=True)

        for param in params:
            comp.add_input(param['name'], param['default'], param['values'])

        for out in outs:
            comp.add_output(out['name'], out['default'], out['values'])

        model.add_subsystem('comp', comp, promotes=["*"])
        self.prob = om.Problem(model)
        self.prob.setup()
        self.prob['x'] = 1.0
        self.prob['y'] = 0.75
        self.prob['z'] = -1.7

    def test_deriv1(self):
        # run at default pt
        self.run_and_check_derivs(self.prob)

        # test output values
        f, g = self.prob['comp.f'], self.prob['comp.g']

        tol = 1e-6
        assert_near_equal(f, -0.05624571, tol)
        assert_near_equal(g, 1.02068754, tol)

    def test_deriv1_swap(self):
        # Bugfix test that we can add outputs before inputs.

        model = om.Group()
        ivc = om.IndepVarComp()

        mapdata = SampleMap()

        params = mapdata.param_data
        x, y, z = params
        outs = mapdata.output_data
        z = outs[0]
        ivc.add_output('x', x['default'], units=x['units'])
        ivc.add_output('y', y['default'], units=y['units'])
        ivc.add_output('z', z['default'], units=z['units'])

        model.add_subsystem('des_vars', ivc, promotes=["*"])

        comp = om.MetaModelStructuredComp(method='scipy_slinear', extrapolate=True)

        for out in outs:
            comp.add_output(out['name'], out['default'], out['values'])

        for param in params:
            comp.add_input(param['name'], param['default'], param['values'])

        model.add_subsystem('comp', comp, promotes=["*"])
        prob = om.Problem(model)
        prob.setup()
        prob['x'] = 1.0
        prob['y'] = 0.75
        prob['z'] = -1.7

        # run at default pt
        self.run_and_check_derivs(prob)

    def test_deriv2(self):
        self.prob['x'] = 10.0
        self.prob['y'] = 0.81
        self.prob['z'] = 1.1
        self.run_and_check_derivs(self.prob)

    def test_deriv3(self):
        self.prob['x'] = 90.0
        self.prob['y'] = 1.2
        self.prob['z'] = 2.1
        self.run_and_check_derivs(self.prob)

    def test_deriv4(self):
        # Tests extrapolation.
        self.prob['x'] = 65.0
        self.prob['y'] = 0.951
        self.prob['z'] = 2.5
        self.run_and_check_derivs(self.prob)

    def test_raise_out_of_bounds_error(self):
        model = om.Group()
        ivc = om.IndepVarComp()

        mapdata = SampleMap()

        params = mapdata.param_data
        x, y, z = params
        outs = mapdata.output_data
        z = outs[0]
        ivc.add_output('x', x['default'], units=x['units'])
        ivc.add_output('y', y['default'], units=y['units'])
        ivc.add_output('z', z['default'], units=z['units'])

        model.add_subsystem('des_vars', ivc, promotes=["*"])

        # Need to make sure extrapolate is False for bounds to be checked
        comp = om.MetaModelStructuredComp(method='scipy_slinear', extrapolate=False)

        for param in params:
            comp.add_input(param['name'], param['default'], param['values'])

        for out in outs:
            comp.add_output(out['name'], out['default'], out['values'])

        model.add_subsystem('comp', comp, promotes=["*"])
        self.prob = om.Problem(model)
        self.prob.setup()

        self.prob['x'] = 1.0
        self.prob['y'] = 0.75
        self.prob['z'] = 9.0 # intentionally set to be out of bounds

        # The interpolating output name is given as a regexp because the exception could
        #   happen with f or g first. The order those are evaluated comes from the keys of
        #   dict so no guarantee on the order except for Python 3.6 !
        msg = "MetaModelStructuredComp \(comp\): Error interpolating output '[f|g]' because input 'comp.z' was " \
              "out of bounds \('.*', '.*'\) with value '9.0'"
        with self.assertRaisesRegex(om.AnalysisError, msg):
            self.run_and_check_derivs(self.prob)

    def test_training_gradient(self):
        model = om.Group()
        ivc = om.IndepVarComp()

        mapdata = SampleMap()

        params = mapdata.param_data
        outs = mapdata.output_data

        ivc.add_output('x', np.array([-0.3, 0.7, 1.2]))
        ivc.add_output('y', np.array([0.14, 0.313, 1.41]))
        ivc.add_output('z', np.array([-2.11, -1.2, 2.01]))

        ivc.add_output('f_train', outs[0]['values'])
        ivc.add_output('g_train', outs[1]['values'])

        comp = om.MetaModelStructuredComp(training_data_gradients=True,
                                          method='scipy_cubic',
                                          vec_size=3)
        for param in params:
            comp.add_input(param['name'], param['default'], param['values'])

        for out in outs:
            comp.add_output(out['name'], out['default'], out['values'])

        model.add_subsystem('ivc', ivc, promotes=["*"])
        model.add_subsystem('comp',
                            comp,
                            promotes=["*"])


        prob = om.Problem(model)
        prob.setup()
        prob.run_model()

        val0 = np.array([ 50.26787317,  49.76106232,  19.66117913])
        val1 = np.array([-32.62094041, -31.67449135, -27.46959668])

        tol = 1e-5
        assert_near_equal(prob['f'], val0, tol)
        assert_near_equal(prob['g'], val1, tol)
        self.run_and_check_derivs(prob)

    def test_training_gradient_setup_called_twice(self):
        model = om.Group()
        ivc = om.IndepVarComp()

        mapdata = SampleMap()

        params = mapdata.param_data
        outs = mapdata.output_data

        ivc.add_output('x', np.array([-0.3, 0.7, 1.2]))
        ivc.add_output('y', np.array([0.14, 0.313, 1.41]))
        ivc.add_output('z', np.array([-2.11, -1.2, 2.01]))

        ivc.add_output('f_train', outs[0]['values'])
        ivc.add_output('g_train', outs[1]['values'])

        comp = om.MetaModelStructuredComp(training_data_gradients=True,
                                          method='scipy_cubic',
                                          vec_size=3)
        for param in params:
            comp.add_input(param['name'], param['default'], param['values'])

        for out in outs:
            comp.add_output(out['name'], out['default'], out['values'])

        model.add_subsystem('ivc', ivc, promotes=["*"])
        model.add_subsystem('comp',
                            comp,
                            promotes=["*"])

        prob = om.Problem(model)
        prob.setup()
        prob.run_model()

        val0 = np.array([ 50.26787317,  49.76106232,  19.66117913])
        val1 = np.array([-32.62094041, -31.67449135, -27.46959668])

        tol = 1e-5
        assert_near_equal(prob['f'], val0, tol)
        assert_near_equal(prob['g'], val1, tol)
        self.run_and_check_derivs(prob)

        # Setup and run again
        prob.setup()
        prob.run_model()

        val0 = np.array([ 50.26787317,  49.76106232,  19.66117913])
        val1 = np.array([-32.62094041, -31.67449135, -27.46959668])

        tol = 1e-5
        assert_near_equal(prob['f'], val0, tol)
        assert_near_equal(prob['g'], val1, tol)
        self.run_and_check_derivs(prob)

    def run_and_check_derivs(self, prob, tol=1e-5, verbose=False):
        """Runs check_partials and compares to analytic derivatives."""

        prob.run_model()
        derivs = prob.check_partials(out_stream=None)

        for i in derivs['comp'].keys():
            if verbose:
                print("Checking derivative pair:", i)
            if derivs['comp'][i]['J_fwd'].sum() != 0.0:
                rel_err = derivs['comp'][i]['rel error'][0]
                self.assertLessEqual(rel_err, tol)

    def test_error_msg_vectorized(self):
        # Tests bug in error message where it doesn't give the correct node value.

        x_bp = np.array([0., 1.])
        y_data = np.array([0., 4.])
        nn = 5

        class MMComp(om.MetaModelStructuredComp):

            def setup(self):
                nn = self.options['vec_size']
                self.add_input(name='x', val=np.ones(nn), units=None, training_data=x_bp)

                self.add_output(name='y', val=np.zeros(nn), units=None, training_data=y_data)

        p = om.Problem()

        ivc = om.IndepVarComp()
        ivc.add_output('x', val=np.linspace(.5, 1.1, nn))

        p.model.add_subsystem('ivc', ivc, promotes=['x'])
        p.model.add_subsystem('MM', MMComp(vec_size=nn), promotes=['x', 'y'])

        p.setup()

        with self.assertRaises(om.AnalysisError) as cm:
            p.run_model()

        msg = ("MMComp (MM): Error interpolating output 'y' because input 'MM.x' was out of bounds ('0.0', '1.0') with value '1.1'")
        self.assertEqual(str(cm.exception), msg)


class TestMetaModelStructuredPython(unittest.TestCase):
    """
    Tests the regular grid map component. specifically the analytic derivatives
    vs. finite difference estimates.
    """

    def setUp(self):

        model = om.Group()
        ivc = om.IndepVarComp()

        mapdata = SampleMap()

        params = mapdata.param_data
        x, y, z = params
        outs = mapdata.output_data
        z = outs[0]
        ivc.add_output('x', x['default'], units=x['units'])
        ivc.add_output('y', y['default'], units=y['units'])
        ivc.add_output('z', z['default'], units=z['units'])

        model.add_subsystem('des_vars', ivc, promotes=["*"])

        comp = om.MetaModelStructuredComp(method='slinear', extrapolate=True)

        for param in params:
            comp.add_input(param['name'], param['default'], param['values'])

        for out in outs:
            comp.add_output(out['name'], out['default'], out['values'])

        model.add_subsystem('comp', comp, promotes=["*"])
        self.prob = om.Problem(model)
        self.prob.setup()
        self.prob['x'] = 1.0
        self.prob['y'] = 0.75
        self.prob['z'] = -1.7

    def run_and_check_derivs(self, prob, tol=1e-5, verbose=False):
        """Runs check_partials and compares to analytic derivatives."""

        prob.run_model()
        derivs = prob.check_partials(method='cs', out_stream=None)

        for i in derivs['comp'].keys():
            if verbose:
                print("Checking derivative pair:", i)
            if derivs['comp'][i]['J_fwd'].sum() != 0.0:
                rel_err = derivs['comp'][i]['rel error'][0]
                self.assertLessEqual(rel_err, tol)

    def test_deriv1(self):
        # run at default pt
        self.run_and_check_derivs(self.prob)

        # test output values
        f, g = self.prob['comp.f'], self.prob['comp.g']

        tol = 1e-6
        assert_near_equal(f, -0.05624571, tol)
        assert_near_equal(g, 1.02068754, tol)

    def test_deriv1_swap(self):
        # Bugfix test that we can add outputs before inputs.

        model = om.Group()
        ivc = om.IndepVarComp()

        mapdata = SampleMap()

        params = mapdata.param_data
        x, y, z = params
        outs = mapdata.output_data
        z = outs[0]
        ivc.add_output('x', x['default'], units=x['units'])
        ivc.add_output('y', y['default'], units=y['units'])
        ivc.add_output('z', z['default'], units=z['units'])

        model.add_subsystem('des_vars', ivc, promotes=["*"])

        comp = om.MetaModelStructuredComp(method='slinear', extrapolate=True)

        for out in outs:
            comp.add_output(out['name'], out['default'], out['values'])

        for param in params:
            comp.add_input(param['name'], param['default'], param['values'])

        model.add_subsystem('comp', comp, promotes=["*"])
        prob = om.Problem(model)
        prob.setup()
        prob['x'] = 1.0
        prob['y'] = 0.75
        prob['z'] = -1.7

        # run at default pt
        self.run_and_check_derivs(prob)

    def test_deriv2(self):
        self.prob['x'] = 10.0
        self.prob['y'] = 0.81
        self.prob['z'] = 1.1
        self.run_and_check_derivs(self.prob)

    def test_deriv3(self):
        self.prob['x'] = 90.0
        self.prob['y'] = 1.2
        self.prob['z'] = 2.1
        self.run_and_check_derivs(self.prob)

    def test_deriv4(self):
        # Tests extrapolation.
        self.prob['x'] = 65.0
        self.prob['y'] = 0.951
        self.prob['z'] = 2.5
        self.run_and_check_derivs(self.prob)

    def test_vectorized_linear(self):
        prob = om.Problem()
        model = prob.model
        ivc = om.IndepVarComp()

        mapdata = SampleMap()

        params = mapdata.param_data
        x, y, _ = params
        outs = mapdata.output_data
        z = outs[0]
        ivc.add_output('x', np.array([x['default'], x['default'], x['default']]),
                       units=x['units'])
        ivc.add_output('y', np.array([y['default'], y['default'], y['default']]),
                       units=x['units'])
        ivc.add_output('z', np.array([z['default'], z['default'], z['default']]),
                       units=x['units'])

        model.add_subsystem('des_vars', ivc, promotes=["*"])

        comp = om.MetaModelStructuredComp(method='slinear', extrapolate=True, vec_size=3)

        for param in params:
            comp.add_input(param['name'], np.array([param['default'], param['default'], param['default']]),
                           param['values'])

        for out in outs:
            comp.add_output(out['name'], np.array([out['default'], out['default'], out['default']]),
                            out['values'])

        model.add_subsystem('comp', comp, promotes=["*"])

        prob.setup(force_alloc_complex=True)
        prob['x'] = np.array([1.0, 10.0, 90.0])
        prob['y'] = np.array([0.75, 0.81, 1.2])
        prob['z'] = np.array([-1.7, 1.1, 2.1])

        prob.run_model()

        partials = prob.check_partials(method='cs', out_stream=None)
        assert_check_partials(partials, rtol=1e-10)

    def test_vectorized_lagrange2(self):
        prob = om.Problem()
        model = prob.model
        ivc = om.IndepVarComp()

        mapdata = SampleMap()

        params = mapdata.param_data
        x, y, _ = params
        outs = mapdata.output_data
        z = outs[0]
        ivc.add_output('x', np.array([x['default'], x['default'], x['default']]),
                       units=x['units'])
        ivc.add_output('y', np.array([y['default'], y['default'], y['default']]),
                       units=x['units'])
        ivc.add_output('z', np.array([z['default'], z['default'], z['default']]),
                       units=x['units'])

        model.add_subsystem('des_vars', ivc, promotes=["*"])

        comp = om.MetaModelStructuredComp(method='lagrange2', extrapolate=True, vec_size=3)

        for param in params:
            comp.add_input(param['name'], np.array([param['default'], param['default'], param['default']]),
                           param['values'])

        for out in outs:
            comp.add_output(out['name'], np.array([out['default'], out['default'], out['default']]),
                            out['values'])

        model.add_subsystem('comp', comp, promotes=["*"])

        prob.setup(force_alloc_complex=True)
        prob['x'] = np.array([1.0, 10.0, 90.0])
        prob['y'] = np.array([0.75, 0.81, 1.2])
        prob['z'] = np.array([-1.7, 1.1, 2.1])

        prob.run_model()

        partials = prob.check_partials(method='cs', out_stream=None)
        # Derivs are large, so ignore atol.
        assert_check_partials(partials, atol=1e10, rtol=1e-10)

    def test_vectorized_lagrange3(self):
        prob = om.Problem()
        model = prob.model

        mapdata = SampleMap()

        params = mapdata.param_data
        x, y, _ = params
        outs = mapdata.output_data

        comp = om.MetaModelStructuredComp(method='lagrange3', extrapolate=True, vec_size=3)

        for param in params:
            comp.add_input(param['name'], np.array([param['default'], param['default'], param['default']]),
                           param['values'], units=param['units'])

        for out in outs:
            comp.add_output(out['name'], np.array([out['default'], out['default'], out['default']]),
                            out['values'])

        model.add_subsystem('comp', comp, promotes=["*"])

        prob.setup(force_alloc_complex=True)
        prob.set_val('x', np.array([1.0, 10.0, 90.0]))
        prob.set_val('y', np.array([0.75, 0.81, 1.2]))
        prob.set_val('z', np.array([-1.7, 1.1, 2.1]))

        prob.run_model()

        partials = prob.check_partials(method='cs', out_stream=None)
        # Derivs are large, so ignore atol.
        assert_check_partials(partials, atol=1e10, rtol=1e-10)

    def test_vectorized_akima(self):
        prob = om.Problem()
        model = prob.model
        ivc = om.IndepVarComp()

        mapdata = SampleMap()

        params = mapdata.param_data
        x, y, _ = params
        outs = mapdata.output_data
        z = outs[0]
        ivc.add_output('x', np.array([x['default'], x['default'], x['default']]),
                       units=x['units'])
        ivc.add_output('y', np.array([y['default'], y['default'], y['default']]),
                       units=y['units'])
        ivc.add_output('z', np.array([z['default'], z['default'], z['default']]),
                       units=z['units'])

        model.add_subsystem('des_vars', ivc, promotes=["*"])

        comp = om.MetaModelStructuredComp(method='akima', extrapolate=True, vec_size=3)

        for param in params:
            comp.add_input(param['name'], np.array([param['default'], param['default'], param['default']]),
                           param['values'])

        for out in outs:
            comp.add_output(out['name'], np.array([out['default'], out['default'], out['default']]),
                            out['values'])

        model.add_subsystem('comp', comp, promotes=["*"])

        prob.setup(force_alloc_complex=True)
        prob['x'] = np.array([1.0, 10.0, 90.0])
        prob['y'] = np.array([0.75, 0.81, 1.2])
        prob['z'] = np.array([-1.7, 1.1, 2.1])

        prob.run_model()

        partials = prob.check_partials(method='cs', out_stream=None)
        # Derivs are large, so ignore atol.
        assert_check_partials(partials, atol=1e10, rtol=1e-10)

    def test_vectorized_cubic(self):
        prob = om.Problem()
        model = prob.model
        ivc = om.IndepVarComp()

        mapdata = SampleMap()

        params = mapdata.param_data
        x, y, _ = params
        outs = mapdata.output_data
        z = outs[0]
        ivc.add_output('x', np.array([x['default'], x['default'], x['default']]),
                       units=x['units'])
        ivc.add_output('y', np.array([y['default'], y['default'], y['default']]),
                       units=x['units'])
        ivc.add_output('z', np.array([z['default'], z['default'], z['default']]),
                       units=x['units'])

        model.add_subsystem('des_vars', ivc, promotes=["*"])

        comp = om.MetaModelStructuredComp(method='cubic', extrapolate=True, vec_size=3)

        for param in params:
            comp.add_input(param['name'], np.array([param['default'], param['default'], param['default']]),
                           param['values'])

        for out in outs:
            comp.add_output(out['name'], np.array([out['default'], out['default'], out['default']]),
                            out['values'])

        model.add_subsystem('comp', comp, promotes=["*"])

        prob.setup(force_alloc_complex=True)
        prob['x'] = np.array([1.0, 10.0, 90.0])
        prob['y'] = np.array([0.75, 0.81, 1.2])
        prob['z'] = np.array([-1.7, 1.1, 2.1])

        prob.run_model()

        partials = prob.check_partials(method='cs', out_stream=None)
        # Derivs are large, so ignore atol.
        assert_check_partials(partials, atol=1e10, rtol=1e-10)

    def test_training_gradient_lagrange3(self):
        model = om.Group()
        ivc = om.IndepVarComp()

        mapdata = SampleMap()

        params = mapdata.param_data
        outs = mapdata.output_data

        ivc.add_output('x', np.array([-0.3, 0.7, 1.2]))
        ivc.add_output('y', np.array([0.14, 0.313, 1.41]))
        ivc.add_output('z', np.array([-2.11, -1.2, 2.01]))

        ivc.add_output('f_train', outs[0]['values'])
        ivc.add_output('g_train', outs[1]['values'])

        comp = om.MetaModelStructuredComp(training_data_gradients=True,
                                          method='lagrange3', vec_size=3)
        for param in params:
            comp.add_input(param['name'], param['default'], param['values'])

        for out in outs:
            comp.add_output(out['name'], out['default'], out['values'])

        model.add_subsystem('ivc', ivc, promotes=["*"])
        model.add_subsystem('comp',
                            comp,
                            promotes=["*"])

        prob = om.Problem(model)
        prob.setup()
        prob.run_model()

        self.run_and_check_derivs(prob)

    def test_training_gradient_akima(self):
        model = om.Group()
        ivc = om.IndepVarComp()

        mapdata = SampleMap()

        params = mapdata.param_data
        outs = mapdata.output_data

        ivc.add_output('x', np.array([-0.3, 0.7, 1.2]))
        ivc.add_output('y', np.array([0.14, 0.313, 1.41]))
        ivc.add_output('z', np.array([-2.11, -1.2, 2.01]))

        ivc.add_output('f_train', outs[0]['values'])
        ivc.add_output('g_train', outs[1]['values'])

        comp = om.MetaModelStructuredComp(training_data_gradients=True,
                                          method='akima', vec_size=3)
        for param in params:
            comp.add_input(param['name'], param['default'], param['values'])

        for out in outs:
            comp.add_output(out['name'], out['default'], out['values'])

        model.add_subsystem('ivc', ivc, promotes=["*"])
        model.add_subsystem('comp',
                            comp,
                            promotes=["*"])

        prob = om.Problem(model)
        prob.setup(force_alloc_complex=True)
        prob.run_model()

        self.run_and_check_derivs(prob)

    def test_training_gradient_akima_basic(self):
        # Mimics usage as an order-reducing interpolating polynomial.
        model = om.Group()
        ivc = om.IndepVarComp()

        mapdata = SampleMap()

        params = mapdata.param_data
        outs = mapdata.output_data

        ivc.add_output('x', np.array([.33]))

        ivc.add_output('f_train', np.array([.3, .7, .5, .6, .3, .4, .2]))

        comp = om.MetaModelStructuredComp(training_data_gradients=True,
                                          method='akima', vec_size=1)
        comp.add_input('x', 0.0, np.array([.1, .2, .3, .4, .5, .6, .7]))
        comp.add_output('f', 0.0, np.array([.3, .7, .5, .6, .3, .4, .2]))

        model.add_subsystem('ivc', ivc, promotes=["*"])
        model.add_subsystem('comp',
                            comp,
                            promotes=["*"])

        prob = om.Problem(model)
        prob.setup(force_alloc_complex=True)
        prob.run_model()

        self.run_and_check_derivs(prob)


@unittest.skipIf(not scipy_gte_019, "only run if scipy>=0.19.")
class TestMetaModelStructuredCompFeature(unittest.TestCase):

    @unittest.skipIf(not scipy_gte_019, "only run if scipy>=0.19.")
    def test_xor(self):
        import numpy as np
        import openmdao.api as om

        # Create regular grid interpolator instance
        xor_interp = om.MetaModelStructuredComp(method='scipy_slinear')

        # set up inputs and outputs
        xor_interp.add_input('x', 0.0, training_data=np.array([0.0, 1.0]), units=None)
        xor_interp.add_input('y', 1.0, training_data=np.array([0.0, 1.0]), units=None)


        xor_interp.add_output('xor', 1.0, training_data=np.array([[0.0, 1.0], [1.0, 0.0]]), units=None)

        # Set up the OpenMDAO model
        model = om.Group()
        model.add_subsystem('comp', xor_interp, promotes=["*"])
        prob = om.Problem(model)
        prob.setup()

        prob.set_val('x', 0)

        # Now test out a 'fuzzy' XOR
        prob.set_val('x', 0.9)
        prob.set_val('y', 0.001242)

        prob.run_model()

        computed = prob.get_val('xor')
        actual = 0.8990064

        assert_almost_equal(computed, actual)

        # we can verify all gradients by checking against finite-difference
        prob.check_partials(compact_print=True)

    @unittest.skipIf(not scipy_gte_019, "only run if scipy>=0.19.")
    def test_shape(self):
        import numpy as np
        import openmdao.api as om

        # create input param training data, of sizes 25, 5, and 10 points resp.
        p1 = np.linspace(0, 100, 25)
        p2 = np.linspace(-10, 10, 5)
        p3 = np.linspace(0, 1, 10)

        # can use meshgrid to create a 3D array of test data
        P1, P2, P3 = np.meshgrid(p1, p2, p3, indexing='ij')
        f = np.sqrt(P1) + P2 * P3

        # verify the shape matches the order and size of the input params
        print(f.shape)

        # Create regular grid interpolator instance
        interp = om.MetaModelStructuredComp(method='scipy_cubic')
        interp.add_input('p1', 0.5, training_data=p1)
        interp.add_input('p2', 0.0, training_data=p2)
        interp.add_input('p3', 3.14, training_data=p3)

        interp.add_output('f', 0.0, training_data=f)

        # Set up the OpenMDAO model
        model = om.Group()
        model.add_subsystem('comp', interp, promotes=["*"])
        prob = om.Problem(model)
        prob.setup()

        # set inputs
        prob.set_val('p1', 55.12)
        prob.set_val('p2', -2.14)
        prob.set_val('p3', 0.323)

        prob.run_model()

        computed = prob.get_val('f')
        actual = 6.73306472

        assert_almost_equal(computed, actual)

        # we can verify all gradients by checking against finite-difference
        prob.check_partials(compact_print=True)

    @unittest.skipIf(not scipy_gte_019, "only run if scipy>=0.19.")
    def test_vectorized(self):
        import numpy as np
        import openmdao.api as om

        # create input param training data, of sizes 25, 5, and 10 points resp.
        p1 = np.linspace(0, 100, 25)
        p2 = np.linspace(-10, 10, 5)
        p3 = np.linspace(0, 1, 10)

        # can use meshgrid to create a 3D array of test data
        P1, P2, P3 = np.meshgrid(p1, p2, p3, indexing='ij')
        f = np.sqrt(P1) + P2 * P3

        # Create regular grid interpolator instance
        interp = om.MetaModelStructuredComp(method='scipy_cubic', vec_size=2)
        interp.add_input('p1', 0.5, training_data=p1)
        interp.add_input('p2', 0.0, training_data=p2)
        interp.add_input('p3', 3.14, training_data=p3)

        interp.add_output('f', 0.0, training_data=f)

        # Set up the OpenMDAO model
        model = om.Group()
        model.add_subsystem('comp', interp, promotes=["*"])
        prob = om.Problem(model)
        prob.setup()

        # set inputs
        prob.set_val('p1', np.array([55.12, 12.0]))
        prob.set_val('p2', np.array([-2.14, 3.5]))
        prob.set_val('p3', np.array([0.323, 0.5]))

        prob.run_model()

        computed = prob['f']
        actual = np.array([6.73306472, 5.2118645])

        assert_almost_equal(computed, actual)

    @unittest.skipIf(not scipy_gte_019, "only run if scipy>=0.19.")
    def test_training_derivatives(self):
        import numpy as np
        import openmdao.api as om

        # create input param training data, of sizes 25, 5, and 10 points resp.
        p1 = np.linspace(0, 100, 25)
        p2 = np.linspace(-10, 10, 5)
        p3 = np.linspace(0, 1, 10)

        # can use meshgrid to create a 3D array of test data
        P1, P2, P3 = np.meshgrid(p1, p2, p3, indexing='ij')
        f = np.sqrt(P1) + P2 * P3

        # verify the shape matches the order and size of the input params
        print(f.shape)

        # Create regular grid interpolator instance
        interp = om.MetaModelStructuredComp(method='scipy_cubic', training_data_gradients=True)
        interp.add_input('p1', 0.5, p1)
        interp.add_input('p2', 0.0, p2)
        interp.add_input('p3', 3.14, p3)

        interp.add_output('f', 0.0, f)

        # Set up the OpenMDAO model
        model = om.Group()
        model.add_subsystem('comp', interp, promotes=["*"])
        prob = om.Problem(model)
        prob.setup()

        # set inputs
        prob.set_val('p1', 55.12)
        prob.set_val('p2', -2.14)
        prob.set_val('p3', 0.323)

        prob.run_model()

        computed = prob.get_val('f')
        actual = 6.73306472

        assert_almost_equal(computed, actual)

        # we can verify all gradients by checking against finite-difference
        prob.check_partials(compact_print=True)

    def test_error_messages_scalar_only(self):
        prob = om.Problem()
        model = prob.model

        comp = om.MetaModelStructuredComp(training_data_gradients=True,
                                          method='slinear', vec_size=3)

        with self.assertRaises(ValueError) as cm:
            comp.add_input('x1', np.array([1.0, 2.0]))

        msg = "MetaModelStructuredComp: Input x1 must either be scalar, or of length equal to vec_size."
        self.assertEqual(str(cm.exception), msg)

        with self.assertRaises(ValueError) as cm:
            comp.add_input('x1', np.zeros((3, 3)))

        self.assertEqual(str(cm.exception), msg)

        with self.assertRaises(ValueError) as cm:
            comp.add_output('x1', np.array([1.0, 2.0]))

        msg = "MetaModelStructuredComp: Output x1 must either be scalar, or of length equal to vec_size."
        self.assertEqual(str(cm.exception), msg)

        with self.assertRaises(ValueError) as cm:
            comp.add_output('x1', np.zeros((3, 3)))

        self.assertEqual(str(cm.exception), msg)


if __name__ == "__main__":
    unittest.main()
