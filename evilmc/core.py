
from astropy.io import ascii
import numpy as np

from .utils import orb_pos, keplereq, cos_f, sin_f

__all__ = ['EVILMC']

freq = np.array([8.6147255e+14, 8.5900415e+14, 8.5654985e+14, 8.5410954e+14, 8.5168309e+14, 8.4927039e+14, 8.4687132e+14, 8.4448577e+14, 8.4211362e+14, 8.3975476e+14, 8.3740907e+14, 8.3507646e+14, 8.3275680e+14, 8.3044999e+14, 8.2815593e+14, 8.2587451e+14, 8.2360563e+14, 8.2134917e+14, 8.1910505e+14, 8.1687316e+14, 8.1465339e+14, 8.1244566e+14, 8.1024986e+14, 8.0806590e+14, 8.0589368e+14, 8.0373310e+14, 8.0158409e+14, 7.9944653e+14, 7.9732034e+14, 7.9520543e+14, 7.9310171e+14, 7.9100910e+14, 7.8892749e+14, 7.8685682e+14, 7.8479698e+14, 7.8274791e+14, 7.8070950e+14, 7.7868168e+14, 7.7666437e+14, 7.7465749e+14, 7.7266095e+14, 7.7067467e+14, 7.6869858e+14, 7.6673260e+14, 7.6477665e+14, 7.6283066e+14, 7.6089454e+14, 7.5896822e+14, 7.5705164e+14, 7.5514471e+14, 7.5324736e+14, 7.5135952e+14, 7.4948112e+14, 7.4761209e+14, 7.4575236e+14, 7.4390186e+14, 7.4206051e+14, 7.4022827e+14, 7.3840504e+14, 7.3659078e+14, 7.3478541e+14, 7.3298887e+14, 7.3120109e+14, 7.2942201e+14, 7.2765157e+14, 7.2588970e+14, 7.2413635e+14, 7.2239144e+14, 7.2065492e+14, 7.1892673e+14, 7.1720681e+14, 7.1549510e+14, 7.1379154e+14, 7.1209608e+14, 7.1040864e+14, 7.0872919e+14, 7.0705766e+14, 7.0539400e+14, 7.0373814e+14, 7.0209004e+14, 7.0044964e+14, 6.9881690e+14, 6.9719174e+14, 6.9557413e+14, 6.9396400e+14, 6.9236131e+14, 6.9076601e+14, 6.8917804e+14, 6.8759736e+14, 6.8602391e+14, 6.8445764e+14, 6.8289851e+14, 6.8134647e+14, 6.7980147e+14, 6.7826346e+14, 6.7673239e+14, 6.7520822e+14, 6.7369089e+14, 6.7218038e+14, 6.7067662e+14, 6.6917957e+14, 6.6768919e+14, 6.6620544e+14, 6.6472827e+14, 6.6325763e+14, 6.6179348e+14, 6.6033579e+14, 6.5888450e+14, 6.5743958e+14, 6.5600098e+14, 6.5456866e+14, 6.5314259e+14, 6.5172271e+14, 6.5030900e+14, 6.4890140e+14, 6.4749989e+14, 6.4610441e+14, 6.4471494e+14, 6.4333143e+14, 6.4195385e+14, 6.4058215e+14, 6.3921631e+14, 6.3785627e+14, 6.3650201e+14, 6.3515349e+14, 6.3381067e+14, 6.3247352e+14, 6.3114200e+14, 6.2981607e+14, 6.2849570e+14, 6.2718085e+14, 6.2587150e+14, 6.2456760e+14, 6.2326912e+14, 6.2197603e+14, 6.2068830e+14, 6.1940588e+14, 6.1812876e+14, 6.1685689e+14, 6.1559024e+14, 6.1432879e+14, 6.1307249e+14, 6.1182132e+14, 6.1057525e+14, 6.0933424e+14, 6.0809827e+14, 6.0686730e+14, 6.0564131e+14, 6.0442026e+14, 6.0320412e+14, 6.0199287e+14, 6.0078647e+14, 5.9958490e+14, 5.9838812e+14, 5.9719611e+14, 5.9600884e+14, 5.9482629e+14, 5.9364841e+14, 5.9247519e+14, 5.9130660e+14, 5.9014261e+14, 5.8898320e+14, 5.8782833e+14, 5.8667798e+14, 5.8553212e+14, 5.8439074e+14, 5.8325379e+14, 5.8212126e+14, 5.8099312e+14, 5.7986934e+14, 5.7874990e+14, 5.7763477e+14, 5.7652394e+14, 5.7541737e+14, 5.7431503e+14, 5.7321692e+14, 5.7212299e+14, 5.7103323e+14, 5.6994762e+14, 5.6886613e+14, 5.6778873e+14, 5.6671540e+14, 5.6564613e+14, 5.6458088e+14, 5.6351964e+14, 5.6246238e+14, 5.6140908e+14, 5.6035972e+14, 5.5931427e+14, 5.5827272e+14, 5.5723503e+14, 5.5620120e+14, 5.5517120e+14, 5.5414501e+14, 5.5312260e+14, 5.5210396e+14, 5.5108906e+14, 5.5007789e+14, 5.4907042e+14, 5.4806663e+14, 5.4706651e+14, 5.4607003e+14, 5.4507718e+14, 5.4408793e+14, 5.4310226e+14, 5.4212016e+14, 5.4114160e+14, 5.4016657e+14, 5.3919505e+14, 5.3822702e+14, 5.3726245e+14, 5.3630134e+14, 5.3534366e+14, 5.3438939e+14, 5.3343852e+14, 5.3249103e+14, 5.3154689e+14, 5.3060610e+14, 5.2966864e+14, 5.2873448e+14, 5.2780361e+14, 5.2687601e+14, 5.2595166e+14, 5.2503056e+14, 5.2411267e+14, 5.2319799e+14, 5.2228649e+14, 5.2137817e+14, 5.2047300e+14, 5.1957097e+14, 5.1867206e+14, 5.1777625e+14, 5.1688353e+14, 5.1599389e+14, 5.1510730e+14, 5.1422375e+14, 5.1334323e+14, 5.1246572e+14, 5.1159121e+14, 5.1071967e+14, 5.0985110e+14, 5.0898548e+14, 5.0812279e+14, 5.0726303e+14, 5.0640616e+14, 5.0555219e+14, 5.0470109e+14, 5.0385285e+14, 5.0300746e+14, 5.0216490e+14, 5.0132516e+14, 5.0048823e+14, 4.9965408e+14, 4.9882271e+14, 4.9799410e+14, 4.9716824e+14, 4.9634511e+14, 4.9552471e+14, 4.9470701e+14, 4.9389201e+14, 4.9307968e+14, 4.9227003e+14, 4.9146303e+14, 4.9065867e+14, 4.8985694e+14, 4.8905783e+14, 4.8826132e+14, 4.8746740e+14, 4.8667605e+14, 4.8588727e+14, 4.8510105e+14, 4.8431736e+14, 4.8353621e+14, 4.8275757e+14, 4.8198143e+14, 4.8120778e+14, 4.8043662e+14, 4.7966792e+14, 4.7890167e+14, 4.7813788e+14, 4.7737651e+14, 4.7661756e+14, 4.7586103e+14, 4.7510689e+14, 4.7435514e+14, 4.7360576e+14, 4.7285875e+14, 4.7211409e+14, 4.7137177e+14, 4.7063179e+14, 4.6989412e+14, 4.6915876e+14, 4.6842570e+14, 4.6769493e+14, 4.6696643e+14, 4.6624020e+14, 4.6551622e+14, 4.6479449e+14, 4.6407500e+14, 4.6335772e+14, 4.6264267e+14, 4.6192981e+14, 4.6121915e+14, 4.6051067e+14, 4.5980437e+14, 4.5910023e+14, 4.5839824e+14, 4.5769839e+14, 4.5700068e+14, 4.5630510e+14, 4.5561162e+14, 4.5492025e+14, 4.5423098e+14, 4.5354379e+14, 4.5285868e+14, 4.5217564e+14, 4.5149465e+14, 4.5081571e+14, 4.5013881e+14, 4.4946394e+14, 4.4879109e+14, 4.4812025e+14, 4.4745141e+14, 4.4678457e+14, 4.4611971e+14, 4.4545683e+14, 4.4479592e+14, 4.4413696e+14, 4.4347995e+14, 4.4282489e+14, 4.4217175e+14, 4.4152054e+14, 4.4087125e+14, 4.4022386e+14, 4.3957837e+14, 4.3893477e+14, 4.3829305e+14, 4.3765321e+14, 4.3701523e+14, 4.3637911e+14, 4.3574484e+14, 4.3511241e+14, 4.3448181e+14, 4.3385304e+14, 4.3322608e+14, 4.3260094e+14, 4.3197759e+14, 4.3135604e+14, 4.3073628e+14, 4.3011829e+14, 4.2950207e+14, 4.2888762e+14, 4.2827493e+14, 4.2766398e+14, 4.2705477e+14, 4.2644729e+14, 4.2584155e+14, 4.2523751e+14, 4.2463520e+14, 4.2403458e+14, 4.2343566e+14, 4.2283843e+14, 4.2224288e+14, 4.2164901e+14, 4.2105681e+14, 4.2046627e+14, 4.1987738e+14, 4.1929014e+14, 4.1870454e+14, 4.1812057e+14, 4.1753823e+14, 4.1695751e+14, 4.1637840e+14, 4.1580090e+14, 4.1522500e+14, 4.1465069e+14, 4.1407797e+14, 4.1350682e+14, 4.1293726e+14, 4.1236925e+14, 4.1180281e+14, 4.1123793e+14, 4.1067459e+14, 4.1011279e+14, 4.0955252e+14, 4.0899379e+14, 4.0843658e+14, 4.0788088e+14, 4.0732670e+14, 4.0677401e+14, 4.0622283e+14, 4.0567314e+14, 4.0512493e+14, 4.0457820e+14, 4.0403295e+14, 4.0348916e+14, 4.0294684e+14, 4.0240597e+14, 4.0186655e+14, 4.0132858e+14, 4.0079204e+14, 4.0025694e+14, 3.9972326e+14, 3.9919101e+14, 3.9866017e+14, 3.9813074e+14, 3.9760272e+14, 3.9707609e+14, 3.9655086e+14, 3.9602701e+14, 3.9550455e+14, 3.9498346e+14, 3.9446375e+14, 3.9394540e+14, 3.9342841e+14, 3.9291278e+14, 3.9239849e+14, 3.9188555e+14, 3.9137395e+14, 3.9086369e+14, 3.9035475e+14, 3.8984714e+14, 3.8934084e+14, 3.8883586e+14, 3.8833219e+14, 3.8782982e+14, 3.8732874e+14, 3.8682897e+14, 3.8633047e+14, 3.8583327e+14, 3.8533734e+14, 3.8484268e+14, 3.8434929e+14, 3.8385717e+14, 3.8336630e+14, 3.8287669e+14, 3.8238833e+14, 3.8190121e+14, 3.8141533e+14, 3.8093068e+14, 3.8044727e+14, 3.7996508e+14, 3.7948411e+14, 3.7900436e+14, 3.7852582e+14, 3.7804848e+14, 3.7757235e+14, 3.7709742e+14, 3.7662368e+14, 3.7615113e+14, 3.7567976e+14, 3.7520957e+14, 3.7474056e+14, 3.7427272e+14, 3.7380604e+14, 3.7334053e+14, 3.7287618e+14, 3.7241298e+14, 3.7195093e+14, 3.7149002e+14, 3.7103026e+14, 3.7057163e+14, 3.7011413e+14, 3.6965777e+14, 3.6920252e+14, 3.6874840e+14, 3.6829539e+14, 3.6784349e+14, 3.6739271e+14, 3.6694302e+14, 3.6649444e+14, 3.6604695e+14, 3.6560055e+14, 3.6515524e+14, 3.6471101e+14, 3.6426786e+14, 3.6382579e+14, 3.6338479e+14, 3.6294485e+14, 3.6250598e+14, 3.6206817e+14, 3.6163142e+14, 3.6119572e+14, 3.6076107e+14, 3.6032746e+14, 3.5989490e+14, 3.5946337e+14, 3.5903287e+14, 3.5860341e+14, 3.5817497e+14, 3.5774755e+14, 3.5732115e+14, 3.5689577e+14, 3.5647140e+14, 3.5604804e+14, 3.5562568e+14, 3.5520432e+14, 3.5478396e+14, 3.5436460e+14, 3.5394622e+14, 3.5352883e+14, 3.5311242e+14, 3.5269700e+14, 3.5228255e+14, 3.5186907e+14, 3.5145656e+14, 3.5104502e+14, 3.5063444e+14, 3.5022482e+14, 3.4981616e+14, 3.4940845e+14, 3.4900169e+14, 3.4859587e+14, 3.4819100e+14, 3.4778706e+14, 3.4738406e+14, 3.4698200e+14, 3.4658086e+14, 3.4618066e+14, 3.4578137e+14, 3.4538300e+14, 3.4498556e+14, 3.4458902e+14, 3.4419340e+14, 3.4379868e+14, 3.4340487e+14, 3.4301195e+14, 3.4261994e+14, 3.4222882e+14, 3.4183860e+14, 3.4144926e+14, 3.4106081e+14, 3.4067324e+14, 3.4028655e+14, 3.3990073e+14, 3.3951580e+14, 3.3913173e+14, 3.3874853e+14, 3.3836619e+14, 3.3798472e+14, 3.3760411e+14, 3.3722435e+14, 3.3684545e+14, 3.3646739e+14, 3.3609019e+14, 3.3571383e+14, 3.3533831e+14, 3.3496363e+14, 3.3458979e+14, 3.3421678e+14, 3.3384460e+14, 3.3347325e+14, 3.3310272e+14, 3.3273302e+14, 3.3236413e+14, 3.3199607e+14, 3.3162881e+14, 3.3126237e+14, 3.3089674e+14, 3.3053192e+14, 3.3016789e+14, 3.2980467e+14, 3.2944225e+14, 3.2908062e+14, 3.2871979e+14, 3.2835975e+14, 3.2800049e+14, 3.2764202e+14, 3.2728433e+14, 3.2692742e+14, 3.2657129e+14, 3.2621594e+14, 3.2586136e+14, 3.2550754e+14, 3.2515450e+14, 3.2480222e+14, 3.2445070e+14, 3.2409994e+14, 3.2374994e+14, 3.2340070e+14, 3.2305221e+14, 3.2270447e+14, 3.2235747e+14, 3.2201122e+14, 3.2166572e+14, 3.2132095e+14, 3.2097693e+14, 3.2063363e+14, 3.2029108e+14, 3.1994925e+14, 3.1960815e+14, 3.1926778e+14, 3.1892814e+14, 3.1858921e+14, 3.1825101e+14, 3.1791352e+14, 3.1757675e+14, 3.1724069e+14, 3.1690534e+14, 3.1657069e+14, 3.1623676e+14, 3.1590353e+14, 3.1557100e+14, 3.1523917e+14, 3.1490803e+14, 3.1457759e+14, 3.1424785e+14, 3.1391879e+14, 3.1359043e+14, 3.1326275e+14, 3.1293575e+14, 3.1260943e+14, 3.1228380e+14, 3.1195884e+14, 3.1163456e+14, 3.1131095e+14, 3.1098802e+14, 3.1066575e+14, 3.1034415e+14, 3.1002321e+14, 3.0970294e+14, 3.0938333e+14, 3.0906438e+14])
resp = np.array([4.5243463e-26, 6.1348948e-26, 6.4561333e-26, 7.9314182e-26, 9.3818933e-26, 9.7262313e-26, 9.0707993e-26, 9.6686057e-26, 1.1752333e-25, 1.3986584e-25, 1.6288096e-25, 1.8227792e-25, 1.9626379e-25, 2.1735205e-25, 2.3997654e-25, 2.6679721e-25, 3.0009023e-25, 3.2929358e-25, 3.5254820e-25, 3.7379608e-25, 3.9029116e-25, 4.0422396e-25, 4.1463752e-25, 4.2514603e-25, 4.3251857e-25, 4.3484710e-25, 4.3858155e-25, 4.6907453e-25, 4.6733536e-25, 4.6792813e-25, 4.6421792e-25, 4.5613634e-25, 4.4602324e-25, 4.3772133e-25, 4.2493284e-25, 4.1052424e-25, 3.9643941e-25, 3.8416853e-25, 3.6678390e-25, 3.4970293e-25, 3.3243041e-25, 3.1143232e-25, 2.9578566e-25, 2.8353561e-25, 2.7473643e-25, 2.5810838e-25, 2.3353302e-25, 2.0713647e-25, 1.9354029e-25, 1.9925055e-25, 2.1135155e-25, 2.2038052e-25, 2.1081252e-25, 1.8344006e-25, 1.6009872e-25, 1.5439535e-25, 1.8020633e-25, 2.1775715e-25, 2.5677235e-25, 2.8290335e-25, 2.9873212e-25, 3.2251785e-25, 3.8745842e-25, 5.4430220e-25, 8.1533532e-25, 1.1834305e-24, 1.5779350e-24, 1.9417450e-24, 2.3032384e-24, 2.8073515e-24, 3.6892287e-24, 5.1826349e-24, 7.5316104e-24, 1.0760066e-23, 1.4910010e-23, 1.9874870e-23, 2.5965633e-23, 3.3920758e-23, 4.4310803e-23, 5.7534215e-23, 7.3935365e-23, 9.2084213e-23, 1.1225032e-22, 1.3198162e-22, 1.5064759e-22, 1.6885694e-22, 1.8785810e-22, 2.0702924e-22, 2.2573677e-22, 2.4269854e-22, 2.5724893e-22, 2.6806764e-22, 2.7639388e-22, 2.8284140e-22, 2.8738392e-22, 2.9130423e-22, 2.9525113e-22, 2.9856421e-22, 3.0189814e-22, 3.0591942e-22, 3.0929814e-22, 3.1202542e-22, 3.1409229e-22, 3.1616829e-22, 3.1825341e-22, 3.2103219e-22, 3.2520121e-22, 3.3008820e-22, 3.3570234e-22, 3.4205285e-22, 3.4774963e-22, 3.5137810e-22, 3.5432246e-22, 3.5586466e-22, 3.5741023e-22, 3.5824408e-22, 3.6051137e-22, 3.6350951e-22, 3.6869644e-22, 3.7464529e-22, 3.8136561e-22, 3.8739939e-22, 3.9347421e-22, 3.9663034e-22, 3.9980254e-22, 4.0149844e-22, 4.0244846e-22, 4.0339575e-22, 4.0660758e-22, 4.0983574e-22, 4.1612878e-22, 4.2246371e-22, 4.2884073e-22, 4.3603153e-22, 4.4172120e-22, 4.4589050e-22, 4.5086431e-22, 4.5429856e-22, 4.5774958e-22, 4.6200864e-22, 4.6708540e-22, 4.7219213e-22, 4.7732892e-22, 4.8329999e-22, 4.8850037e-22, 4.9292033e-22, 4.9655011e-22, 4.9856242e-22, 5.0139946e-22, 5.0424720e-22, 5.0793293e-22, 5.1329716e-22, 5.2036003e-22, 5.2662977e-22, 5.3293719e-22, 5.3928226e-22, 5.4481791e-22, 5.4953403e-22, 5.5085851e-22, 5.5218054e-22, 5.5177848e-22, 5.5049617e-22, 5.5005854e-22, 5.5047578e-22, 5.5175790e-22, 5.5479319e-22, 5.5960200e-22, 5.6532034e-22, 5.7107046e-22, 5.7685234e-22, 5.8266618e-22, 5.8761349e-22, 5.9168404e-22, 5.9486734e-22, 5.9806199e-22, 6.0126798e-22, 6.0540130e-22, 6.0955297e-22, 6.1372311e-22, 6.1791162e-22, 6.2304869e-22, 6.2634434e-22, 6.2965162e-22, 6.3202987e-22, 6.3346861e-22, 6.3490471e-22, 6.3538694e-22, 6.3681415e-22, 6.3823869e-22, 6.4158426e-22, 6.4590702e-22, 6.5121759e-22, 6.5752691e-22, 6.6386957e-22, 6.6926576e-22, 6.7468815e-22, 6.7914977e-22, 6.8164893e-22, 6.8315824e-22, 6.8466493e-22, 6.8416549e-22, 6.8565929e-22, 6.8715038e-22, 6.8965138e-22, 6.9418972e-22, 7.0078741e-22, 7.0639553e-22, 7.1203040e-22, 7.1872323e-22, 7.2545037e-22, 7.2909619e-22, 7.3275412e-22, 7.3642416e-22, 7.3800680e-22, 7.3958665e-22, 7.4116365e-22, 7.4273784e-22, 7.4537401e-22, 7.4801480e-22, 7.5280512e-22, 7.5869129e-22, 7.6352499e-22, 7.6946232e-22, 7.7542724e-22, 7.8032836e-22, 7.8415442e-22, 7.8579477e-22, 7.8853511e-22, 7.8906687e-22, 7.9069852e-22, 7.9009846e-22, 7.9171649e-22, 7.9333153e-22, 7.9606954e-22, 8.0107191e-22, 8.0609465e-22, 8.1341288e-22, 8.1962553e-22, 8.2586645e-22, 8.3098621e-22, 8.3497316e-22, 8.3897290e-22, 8.4066293e-22, 8.4118492e-22, 8.4052686e-22, 8.3985003e-22, 8.3915410e-22, 8.3962017e-22, 8.4007503e-22, 8.4170759e-22, 8.4452973e-22, 8.4975020e-22, 8.5739316e-22, 8.6266325e-22, 8.6674520e-22, 8.7205288e-22, 8.7738148e-22, 8.8028913e-22, 8.8320156e-22, 8.8366075e-22, 8.8164244e-22, 8.7959655e-22, 8.7628152e-22, 8.7168538e-22, 8.6829434e-22, 8.6486705e-22, 8.6266101e-22, 8.6295001e-22, 8.6322714e-22, 8.6476231e-22, 8.6756768e-22, 8.7293386e-22, 8.7960319e-22, 8.8501565e-22, 8.9173973e-22, 8.9849340e-22, 9.0267889e-22, 9.0687738e-22, 9.1108894e-22, 9.1137938e-22, 9.1165776e-22, 9.1060431e-22, 9.0953020e-22, 9.0710747e-22, 9.0598786e-22, 9.0484748e-22, 9.0234554e-22, 9.0250433e-22, 9.0265065e-22, 9.0278433e-22, 9.0562103e-22, 9.0846217e-22, 9.1267405e-22, 9.1689902e-22, 9.2113691e-22, 9.2538780e-22, 9.3103523e-22, 9.3392887e-22, 9.3821906e-22, 9.4112605e-22, 9.4263676e-22, 9.4273848e-22, 9.4423661e-22, 9.4431689e-22, 9.4580223e-22, 9.4728336e-22, 9.4733345e-22, 9.4737060e-22, 9.4883009e-22, 9.4884536e-22, 9.5029185e-22, 9.5318244e-22, 9.5462447e-22, 9.5751947e-22, 9.5895705e-22, 9.6185637e-22, 9.6328934e-22, 9.6766818e-22, 9.6910102e-22, 9.7349750e-22, 9.7344174e-22, 9.7635845e-22, 9.7778215e-22, 9.8070305e-22, 9.8212209e-22, 9.8353642e-22, 9.8494608e-22, 9.8635112e-22, 9.8622728e-22, 9.8914723e-22, 9.8900481e-22, 9.9038657e-22, 9.9022111e-22, 9.9158868e-22, 9.9295150e-22, 9.9119740e-22, 9.9254135e-22, 9.9231536e-22, 9.9207529e-22, 9.9182107e-22, 9.9155281e-22, 9.8968680e-22, 9.9097361e-22, 9.9225540e-22, 9.9193475e-22, 9.9320171e-22, 9.9446368e-22, 9.9410914e-22, 9.9697199e-22, 9.9821841e-22, 1.0010848e-21, 1.0055851e-21, 1.0068298e-21, 1.0080695e-21, 1.0093038e-21, 1.0105328e-21, 1.0101034e-21, 1.0113172e-21, 1.0125256e-21, 1.0120613e-21, 1.0149263e-21, 1.0127651e-21, 1.0139425e-21, 1.0117420e-21, 1.0095170e-21, 1.0106590e-21, 1.0083944e-21, 1.0078105e-21, 1.0072114e-21, 1.0048824e-21, 1.0042481e-21, 1.0035987e-21, 1.0046630e-21, 1.0022538e-21, 1.0032971e-21, 1.0043344e-21, 1.0053662e-21, 1.0063921e-21, 1.0074122e-21, 1.0066633e-21, 1.0094346e-21, 1.0086643e-21, 1.0078782e-21, 1.0106414e-21, 1.0098339e-21, 1.0108026e-21, 1.0135625e-21, 1.0109202e-21, 1.0118659e-21, 1.0146174e-21, 1.0155559e-21, 1.0164883e-21, 1.0155879e-21, 1.0165031e-21, 1.0155756e-21, 1.0164734e-21, 1.0155186e-21, 1.0163990e-21, 1.0154167e-21, 1.0162795e-21, 1.0152697e-21, 1.0161148e-21, 1.0150773e-21, 1.0121420e-21, 1.0129530e-21, 1.0137574e-21, 1.0107627e-21, 1.0115441e-21, 1.0123191e-21, 1.0092645e-21, 1.0100162e-21, 1.0088397e-21, 1.0076464e-21, 1.0064365e-21, 1.0071464e-21, 1.0039659e-21, 1.0027052e-21, 1.0014276e-21, 1.0001329e-21, 9.9882133e-22, 9.9355778e-22, 9.9417431e-22, 9.9280617e-22, 9.9142087e-22, 9.9001837e-22, 9.8859857e-22, 9.8516315e-22, 9.8370344e-22, 9.8222644e-22, 9.8475950e-22, 9.8325789e-22, 9.8376300e-22, 9.8223159e-22, 9.8475170e-22, 9.8319544e-22, 9.8162146e-22, 9.8413045e-22, 9.8458709e-22, 9.8503638e-22, 9.8547850e-22, 9.8591328e-22, 9.8841715e-22, 9.9092424e-22, 9.8926041e-22, 9.8967087e-22, 9.9007406e-22, 9.9046974e-22, 9.8874982e-22, 9.8701180e-22, 9.8525561e-22, 9.8135716e-22, 9.9872451e-22, 9.9909120e-22, 9.9517008e-22, 9.8907444e-22, 9.8724076e-22, 9.8323257e-22, 9.7703357e-22, 9.7296164e-22, 9.7103284e-22, 9.6908540e-22, 9.6493627e-22, 9.6294618e-22, 9.6093745e-22, 9.5890997e-22, 9.5465903e-22, 9.5258861e-22, 9.5049936e-22, 9.4839113e-22, 9.4626412e-22, 9.4188607e-22, 9.3971551e-22, 9.3304023e-22, 9.2857218e-22, 9.2407410e-22, 9.1954588e-22, 9.1272258e-22, 9.0812825e-22, 9.0577953e-22, 8.9884861e-22, 8.9416314e-22, 8.8715466e-22, 8.8010455e-22, 8.7531614e-22, 8.6818804e-22, 8.6333258e-22, 8.5380607e-22, 8.4887776e-22, 8.4158713e-22, 8.3892796e-22, 8.3390648e-22, 8.3120186e-22, 8.2847716e-22, 8.2337319e-22, 8.2060279e-22, 8.1781215e-22, 8.1500131e-22, 8.1217029e-22, 8.0931896e-22, 8.0884042e-22, 8.0115672e-22, 8.0064306e-22, 8.0012016e-22, 7.9958830e-22, 7.9420456e-22, 7.9364285e-22, 7.9307207e-22, 7.9005357e-22, 7.8701446e-22, 7.8640452e-22, 7.8087407e-22, 7.8269531e-22, 7.7958468e-22, 7.7398039e-22, 7.7082222e-22, 7.6764311e-22, 7.6444317e-22, 7.6122205e-22, 7.5797993e-22, 7.5471676e-22, 7.5143242e-22, 7.4812693e-22, 7.4480026e-22, 7.4145234e-22, 7.3554673e-22, 7.2960815e-22, 7.2108854e-22, 7.1252423e-22, 7.0135526e-22, 6.8499869e-22, 6.7113341e-22, 6.4946778e-22, 6.2769825e-22, 6.0323573e-22, 5.7606233e-22, 5.4616015e-22, 5.1351137e-22, 4.8071057e-22, 4.4513905e-22, 4.0677873e-22, 3.7087225e-22, 3.3216462e-22, 2.9856421e-22, 2.6136767e-22, 2.2824827e-22, 1.9684027e-22, 1.6848895e-22, 1.4268116e-22, 1.1943460e-22, 9.9840591e-23, 8.2310156e-23, 6.7127127e-23, 5.4847947e-23, 4.4950752e-23, 3.6637528e-23, 2.9919028e-23, 2.4424416e-23, 1.9888767e-23, 1.6263806e-23, 1.3391021e-23, 1.1000464e-23, 9.1229457e-24, 7.5685494e-24, 6.4224989e-24, 5.2713591e-24, 4.4487791e-24, 3.8454823e-24, 3.2674379e-24, 2.8547654e-24, 2.4066077e-24, 2.1307568e-24, 1.8677719e-24, 1.6572692e-24, 1.5363791e-24, 1.3752539e-24, 1.2418358e-24, 1.1932660e-24, 1.0702717e-24, 9.8964031e-25, 8.9718728e-25, 9.5083080e-25, 8.8954760e-25, 8.1068385e-25, 7.6616900e-25, 7.5332868e-25, 7.5494607e-25, 6.8381861e-25, 6.4737438e-25, 6.0784708e-25, 5.7693211e-25, 6.6327472e-25, 5.9116209e-25, 6.3957981e-25, 6.6457054e-25, 5.9494550e-25, 5.3391880e-25, 5.4991429e-25, 4.8554619e-25, 4.9552967e-25, 4.6367210e-25, 4.6764960e-25, 4.2057144e-25, 4.4554158e-25, 4.4648009e-25, 4.5346573e-25, 4.4835998e-25, 4.4019392e-25, 4.2590634e-25, 4.2375017e-25, 4.1241738e-25, 4.9899702e-25, 4.1107526e-25, 4.7341553e-25, 4.0355001e-25, 3.7969473e-25, 4.1451162e-25, 4.0917333e-25, 4.2866007e-25, 4.1087292e-25, 3.8989017e-25, 4.0319924e-25, 3.6644831e-25, 3.7975907e-25])

freq = freq[::-1]
resp = resp[::-1]
sum = np.trapz(resp, freq)

def BBrad_kep_dop(temp, vz):
    #   common arrs, freq, del_freq, resp, sum
    # ; common arrs, wave, del_wave, resp, sum
    #   ;2011 Sep 15 - Returns black body radiation convolved with the Kepler response
    #   ;  function and Doppler shifts
    #   ;
    #   ;2012 Jan 11 - Using expression from Loeb & Gaudi (2003) ApJL 588, L117.
    #   ;
    #   ;Input
    #   ;temp - black body temperature
    #   ;vz - Doppler velocity in units of c
    planck = 6.626068e-34
    c = 299792458.
    boltz = 1.3806503e-23

    if type(temp) is not list:
        temp = [temp]

    F_nu0 = np.zeros(len(freq))
    ret = np.zeros(len(temp))

    freq0 = freq*(1.+vz)
    for i in range(len(temp)):
        x0 = planck*freq0/(boltz*temp[i])
        #From Loeb & Gaudi (2003) ApJL 588, L117, Eqn 3
        alpha0 = (np.exp(x0)*(3.-x0)-3.)/(np.exp(x0)-1.)

        F_nu0 = 2.*planck*(freq0*freq0*freq0)/(c*c)/(np.exp(x0)-1.)
        F_nu = F_nu0*(1.-(3.-alpha0)*vz)

        func = F_nu*resp/sum
        ret[i] = np.trapz(func, freq)

    return ret

def dBBrad_kep_dopdT(temp0, vz):
    # ;2011 Sep 15 - Returns temp derivativ of black body radiation convolved
    # ;  with the Kepler response function and Doppler shifts
    # ;
    # ;2012 Jan 11 - Using expression from Loeb & Gaudi (2003) ApJL 588, L117.
    # ;  Also I'm cheating to calculate the derivative numerically.
    # ;
    # ;Input
    # ;temp - black body temperature
    # ;vz - Doppler velocity in units of c

    del_temp = 1-5*temp0
    BB = BBrad_kep_dop([temp0-0.5*del_temp, temp0+0.5*del_temp], vz)
    ret = (BB[1]-BB[0])/del_temp
    return ret


def nrm(x, y, z):
    #Returns 3-D vector magnitude
    return np.sqrt(x**2+y**2+z**2)


def del_R(q, a, cos_psi, nrm_Omega, cos_lambda):
    # ;2011 Oct 6 - Returns the deformation for a very slightly tidally
    # ;  deformed and slowly rotating body with a Love number of 1
    # ;
    # ;See Eqn (6) from Jackson+ (2012) ApJ.

    return (q*(1./np.sqrt(a*a-2.*a*cos_psi+1.) - 1./np.sqrt(a*a+1) - cos_psi/(a*a))
            - nrm_Omega*nrm_Omega/(2.*a*a*a)*(cos_lambda*cos_lambda))


def del_gam_vec(del_R, rhat, q, a, ahat, cos_psi, nrm_Omega, Omega_hat, cos_lambda):
    # ;2011 Oct 6 - Returns the small correction to the surface gravity vector for a very
    # ;  slightly tidally deformed and slowly rotating body
    # ;
    # ;See Eqn (8) from Jackson+ (2012) ApJ.

    term1 = np.sqrt(a*a - 2.*a*cos_psi + 1.)

    return (2.*del_R*rhat + q*(a*ahat-rhat)/(term1*term1*term1)
            + nrm_Omega*nrm_Omega/(a*a*a)*(rhat-Omega_hat*cos_lambda) - q/(a*a)*ahat)


def rhat_dot_del_gam0(q, a, nrm_Omega):
    # ;2011 Oct 6 - Returns the small correction to the magnitude of the surface gravity
    # ;  vector at the "pole" (see Jackson et al.) for a very slightly tidally deformed
    # ;  and slowly rotating body
    # ;
    # ;See Eqn (9) from Jackson+ (2012) ApJ.

    term1 = np.sqrt(a*a+1.)

    return -q/(term1*term1*term1)+nrm_Omega*nrm_Omega/(a*a*a)


def del_temp(bet, rhat_dot_dgam, dgam0):
    # ;2011 Oct 6 - Returns the small correction to surface temp for a star with a
    # ;  small amount of gravity darkening
    # ;
    # ;See Eqn (10) from Jackson+ (2012) ApJ.

    return bet*(dgam0-rhat_dot_dgam)

def LDC(LDC_law, gam, mu):
    # ;2011 Jul 25 - Returns limb-darkened flux
    ret = 1-gam[0]*(1-mu)-gam[1]*(1-mu)**2
    return ret


def EVILMC(phs, params):
    # ;+
    # ; NAME:
    # ;	EVILMC
    # ;
    # ; PURPOSE:
    # ;	This function returns the ellipsoidal variation of a slowly-rotating star induced by a
    # ;	low-mass companion. This code is publicly available with NO warranties whatsoever at
    # ;	http://www.lpl.arizona.edu/~bjackson/code/idl.html. If you use the code, please cite the
    # ;	following paper: Jackson et al. (2012) ApJ 750, 1.
    # ;
    # ; CATEGORY:
    # ;	Astrophysics.
    # ;
    # ; CALLING SEQUENCE:
    # ;
    # ;	Result = EVILMC(phs, params)
    # ;
    # ; INPUTS:
    # ;	phs:	orbital phase; phs = 0 corresponds to inferior conjunction (i.e. mid-transit)
    # ;		that positional parameters are shown with Initial Caps.
    # ;	params: array of ellipsoidal variation parameters, as follows --
    # ;  		params[0] - # of lat/long grid points on star,
    # ;  		params[1] - mass ratio, q (= M_p/M_*)
    # ;		params[2] - K_z, stellar reflex velocity, in m/s
    # ;		params[3] - T_0, stellar effective temp in K
    # ;		params[4:6] - x, y, z of stellar rotation vector in units of mean motion
    # ;		params[7:8] - \gamma_i, stellar quadratic limb-darkening coefficients
    # ;		params[9] - \beta, stellar gravity darkening exponent
    # ;		params[10] - a, semi-major axis (in units of stellar radii)
    # ;		params[11] - orbital period (in days)
    # ;		params[12] - orbital inclination in degrees
    # ;		params[13] - eccentricity (assumed 0 for now)
    # ;		params[14] - longitude of ascending node (should probably be 0)
    # ;		params[15] - longitude of pericenter (while ecc = 0, this is assumed 0)
    # ;
    # ; OUTPUTS:
    # ;	Stellar disk emission in MKS units - Typically, the disk emission should be normalized
    # ;	by the disk emission at phs = 0.5, when a planet is eclipse by the star. See the
    # ;	companion code, EVILMC_plphs.pro to see how this is done.
    # ;
    # ; RESTRICTIONS:
    # ;	No checking of parameters is done, so be sure everything is in the correct units, etc.
    # ;	Also, the code requires companion routines: orb_pos.pro.
    # ;	These are available at http://www.lpl.arizona.edu/~bjackson/code/idl.html. Also, for now
    # ;	the code assumes an orbital eccentricity of 0 and blackbody radiation from the star (among
    # ;	other important assumptions).
    # ;
    # ; EXAMPLE:
    # ;	;See the companion code EVILMC_plphs.pro and example_EVILMC.pro for more help. Both are
    # ;	;available at http://www.lpl.arizona.edu/~bjackson/code/idl.html.
    # ;
    # ;	q = 1.10d-3
    # ;	semi = 4.15 ;A/R_0
    # ;	per = 2.204733 ;days
    # ;	Omega = 4.73d-7 ;s^{-1}
    # ;	Ts = 6350. ;K
    # ;	Kz = 300. ;m/s
    # ;	logg = 4.07 ;log(cm/s^2)
    # ;	M = 0.26 ;[Fe/H]
    # ;	ecc = 0.0 ;orbital eccentricity
    # ;	asc_node = 0. ;longitude of planetary ascending node
    # ;	peri_long = 0. ;longitude of planetary pericenter
    # ;	inc = 83.1 ;orbital inclination in degrees
    # ;	bet = 0.0705696 ;gravity darkening exponent, (T/T_0) = (g/g_0)^\beta
    # ;	;limb-darkening coefficients, I(\mu)/I(1) = 1 - \gamma_1 (1-\mu) - \gamma_2 (1-\mu)^2
    # ;	gam1 = 0.314709
    # ;	gam2 = 0.312125
    # ;	gam = [gam1, gam2]
    # ;	;number of latitude or longitude grid points on stellar surface
    # ;	num_grid = 31
    # ;	;orientation (X, Y, Z) of stellar rotation axis
    # ;	Omegahat = [0., 0., 1.0]
    # ;	;stellar rotation axis
    # ;	Omega = Omega*Omegahat
    # ;	p = [num_grid, q, Kz, Ts, Omega, gam, bet, semi, per, inc, ecc, asc_node, peri_long]
    # ;
    # ;	;orbital phase
    # ;	num_phs = 101
    # ;	phs = dindgen(num_phs)*1./(num_phs-1.)
    # ;	ret = EVILMC(phs, p)
    # ;
    # ;	;calculate normalization
    # ;	norm_st_phs = EVILMC(0.5, p)
    # ;	ret /= norm_st_phs
    # ;	plot, phs, ret, yr=[min(ret), max(ret)]
    # ;
    # ; MODIFICATION HISTORY:
    # ; 	Written by:	Brian Jackson (decaelus@gmail.com, 2011 Oct 5.
    # ;-

    #;MKS speed of light
    c = 299792458.

    #;assigning calculation parameters
    num_grid = params[0]
    q = params[1]
    Kz = params[2] #;reflex velocity amplitude, m/s
    Ts = params[3]
    Omega = params[4]
    gam = params[5] #;assuming quadratic limb-darkening
    bet = params[6]
    semi = params[7]
    per = params[8]*86400. #;Convert from days to seconds
    inc = params[9]*np.pi/180.
    ecc = params[10]
    asc_node = params[11]
    peri_long = params[12]

    #;Save some time by doing all the sine and cosine calculations once
    sin_asc = np.sin(asc_node)
    cos_asc = np.cos(asc_node)
    sin_inc = np.sin(inc)
    cos_inc = np.cos(inc)
    cos_peri = np.cos(peri_long)
    sin_peri = np.sin(peri_long)

    #;Given the way the coordinate system is defined, this is the relationship
    #;  between the orbital mean anomaly and phase
    mean_anom = (phs+0.25)*2.*np.pi

    #;Calculate 3-D orbital position
    rc = orb_pos(semi, ecc, asc_node, peri_long, inc, mean_anom)
    nrm_rc = nrm(rc[0,:], rc[1,:], rc[2,:])
    cos_f_i = cos_f(mean_anom, ecc)
    sin_f_i = sin_f(mean_anom, ecc)
    #;cos(peri_long + f)
    cos_wf = cos_peri*cos_f_i-sin_peri*sin_f_i
    sin_wf = sin_peri*cos_f_i+cos_peri*sin_f_i

    #;stellar Doppler velocity in units of c
    vz = Kz/c*cos_wf

    #;make grid on stellar surface
    dcos_theta = 1./(num_grid)
    cos_theta = (np.arange(num_grid)*dcos_theta+0.5*dcos_theta)[::-1]
    sin_theta = np.sqrt(1.-cos_theta**2.)

    dphi = 2.*np.pi/(num_grid)
    phi = np.arange(num_grid)*dphi+0.5*dphi
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    xhat = cos_phi * sin_theta
    yhat = sin_phi * sin_theta
    zhat = np.ones(len(sin_phi)) * cos_theta

    nrm_Omega = nrm(Omega[0], Omega[1], Omega[2])
    Omegahat = Omega/nrm_Omega
    #;If nrm_Omega eq 0, then assumed Omegahat is arbitrary
    if nrm_Omega == 0.:
        Omegahat = [0., 0., 1.0]
    cos_lambda = Omegahat[0]*xhat+Omegahat[1]*yhat+Omegahat[2]*zhat

    disk = np.zeros(len(vz))

    BBrad0 = BBrad_kep_dop(Ts, 0.)
    dBB0 = dBBrad_kep_dopdT(Ts, 0.)
    if (min(abs(vz)) > 0.):
        if(len(phs) < 2):
            BBrad0 = BBrad_kep_dop(Ts, vz)
            dBB0 = dBBrad_kep_dopdT(Ts, vz)
        else:
            #;Generate BB radiation array for interpolation
            mn = min(vz)
            mx = max(vz)

            desired_frac = 0.1 #;gives convergence to better than 1 in 10^15 typically
            num = np.floor(abs(mx-mn)/abs(desired_frac*mn) + 1)
            vz_arr = np.arange(num)*(mx-mn)/(num-1.) + mn
            BBrad0 = np.zeros(len(vz_arr))
            dBB0 = np.zeros(len(vz_arr))

            for i in range(0, int(num)):
                BBrad0[i] = BBrad_kep_dop(Ts, vz_arr[i])
                dBB0[i] = dBBrad_kep_dopdT(Ts, vz_arr[i])

    for i in range(0, len(vz)):
        BBrad = BBrad0
        dBB = dBB0

        if ((len(phs) > 1) and (min(abs(vz)) > 0.)):
            #;blackbody radiation at Ts integrated over Kepler bandpass

            BBrad = np.interp(vz[i], BBrad0, vz_arr)
            dBB = np.interp(vz[i], dBB0, vz_arr)

        cos_psi = rc[0,i]/nrm_rc[i]*xhat+rc[1,i]/nrm_rc[i]*yhat+rc[2,i]/nrm_rc[i]*zhat

        del_R_i = del_R(q, nrm_rc[i], cos_psi, nrm_Omega, cos_lambda)

        del_gam_vec_x = del_gam_vec(del_R_i, xhat, q, semi, rc[0,i]/nrm_rc[i], cos_psi, nrm_Omega, Omegahat[0], cos_lambda)
        del_gam_vec_y = del_gam_vec(del_R_i, yhat, q, semi, rc[1,i]/nrm_rc[i], cos_psi, nrm_Omega, Omegahat[1], cos_lambda)
        del_gam_vec_z = del_gam_vec(del_R_i, zhat, q, semi, rc[2,i]/nrm_rc[i], cos_psi, nrm_Omega, Omegahat[2], cos_lambda)
        gx = -xhat + del_gam_vec_x
        gy = -yhat + del_gam_vec_y
        gz = -zhat + del_gam_vec_z

        rhat_dot_dgam = xhat*del_gam_vec_x + yhat*del_gam_vec_y + zhat*del_gam_vec_z
        nrm_g = 1. - rhat_dot_dgam

        mu = abs(gz)/nrm_g

        dgam0 = rhat_dot_del_gam0(q, nrm_rc[i], nrm_Omega)
        nrm_ogam0 = 1. - dgam0

        dtemp = del_temp(bet, rhat_dot_dgam, dgam0)*Ts
        temp = Ts + dtemp

        #;stellar radiation at temp
        # BB = (BBrad * np.ones(len(dtemp[:,0]), len(dtemp[0,:])) +
        #       dBB * np.ones(len(dtemp[:,0]), len(dtemp[0,:]))*dtemp)
        BB = (BBrad * np.ones((len(dtemp), len(dtemp))) +
              dBB * np.ones((len(dtemp), len(dtemp)))*dtemp)

        #;limb-darkening profile
        prof = LDC(1, gam, mu)

        #;projected area of each grid element
        dareap = (1.+2.*del_R_i)*mu*dcos_theta*dphi

        #;integrated disk brightness
        disk[i] = np.sum(prof*BB*dareap)
        # import matplotlib.pyplot as plt
        # plt.imshow(prof*BB*dareap)
        # plt.show()

    return disk


def EVILMC_plphs(phs, params):
    # ; NAME:
    # ;	EVILMC_plphs
    # ;
    # ; PURPOSE:
    # ;	Calls routines to calculate stellar ellipsoidal variation and planetary phase function.
    # ;	Code is provided with no warranties whatsoever at
    # ;	http://www.lpl.arizona.edu/~bjackson/code/idl.html. See also paper Jackson et al. (2012)
    # ;	ApJ 750, 1 for more details.
    # ;
    # ; CATEGORY:
    # ;	Astrophysics.
    # ;
    # ; CALLING SEQUENCE:
    # ;	Result = EVILMC_plphs(phs, params)
    # ;
    # ; INPUTS:
    # ;       phs:    orbital phase; phs = 0 corresponds to inferior conjunction (i.e. mid-transit)
    # ;               that positional parameters are shown with Initial Caps.
    # ;       params: array of ellipsoidal variation parameters, as follows --
    # ;               params[0] - # of lat/long grid points on star,
    # ;               params[1] - mass ratio, q (= M_p/M_*)
    # ;               params[2] - K_z, stellar reflex velocity, in m/s
    # ;               params[3] - T_0, stellar effective temp in K
    # ;               params[4:6] - x, y, z of stellar rotation vector in units of mean motion
    # ;               params[7:8] - \gamma_i, stellar quadratic limb-darkening coefficients
    # ;               params[9] - \beta, stellar gravity darkening exponent
    # ;               params[10] - a, semi-major axis (in units of stellar radii)
    # ;               params[11] - orbital period (in days)
    # ;               params[12] - orbital inclination in degrees
    # ;               params[13] - eccentricity (assumed 0 for now)
    # ;               params[14] - longitude of ascending node (should probably be 0)
    # ;               params[15] - longitude of pericenter (while ecc = 0, this is assumed 0)
    # ;               params[16] - F_0, planetary phase function parameter
    # ;               params[17] - F_1, planetary phase function parameter
    # ;
    # ; OUTPUTS:
    # ;	Normalized stellar ellipsoidal variation plus planetary phase function
    # ;
    # ; RESTRICTIONS:
    # ;	No checking of parameters is done, so be sure everything is in the correct units, etc.
    # ;	Also, the code requires a few companion routines: EVILMC.pro.
    # ;	These are available at http://www.lpl.arizona.edu/~bjackson/code/idl.html.
    # ;
    # ; EXAMPLE:
    # ;
    # ;       See the companion code example_EVILMC.pro to see how to use this code.
    # ;
    # ; MODIFICATION HISTORY:
    # ; 	Written by:	Brian Jackson, 2012 April 18.
    # ;-
    EV_params = params[0:-2]
    pl_params = params[-2:]

    pl_phs = pl_phs_crv(phs, pl_params[0], pl_params[1])

    st_phs = EVILMC(phs, EV_params)
    # ;calculate the normalization
    norm_st_phs = EVILMC(np.array([0.5]), EV_params)
    # ;normalize
    st_phs /= norm_st_phs #* np.ones_like(phs)

    ret = st_phs + pl_phs

    return ret

def pl_phs_crv(phs, F0, F1):
# ;+
# ; NAME:
# ;	pl_phs_crv
# ;
# ; PURPOSE:
# ;	Returns a sinusoidal planetary phase curve for a planet in
# ; 	a circular orbit
# ;
# ; CATEGORY:
# ;	Astrophysics
# ;
# ; CALLING SEQUENCE:
# ;
# ;	pl_phs_crv = pl_phs_crv(phs, F0, F1)
# ;
# ; INPUTS:
# ;	phs: 	orbital phase (0 = mid-transit, 0.5 = mid-eclipse)
# ;	F0/F1:	phase curve = F0 - F1*cos(2*!pi*phs)
# ;
# ; OUTPUTS:
# ;	planetary phase curve
# ;
# ; MODIFICATION HISTORY:
# ; 	2012 Sep 21 -- Written by Brian Jackson (decaelus@gmail.com)
# ;-

    ret = F0 - F1*np.cos(2.*np.pi*phs)

    return ret
