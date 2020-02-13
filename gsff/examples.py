#!/usr/bin/env python3

"""Examples for gsff.main

https://github.com/schwanbeck/GSFF

Copyright (c) 2020 Julian Schwanbeck (julian.schwanbeck@med.uni-goettingen.de)
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

example_a = [
    # (x-Position, y-Position)
    (84.5, 908.0,),
    (84.00000762939453, 907.5001220703124,),
    (83.25000762939453, 906.7501220703124,),
    (83.00000762939453, 906.5001220703124,),
    (83.00000762939453, 906.5001220703124,),
    (82.4999771118164, 906.0001220703124,),
    (82.00000762939453, 905.5001220703124,),
    (81.50005340576172, 905.5,),
    (80.99999237060547, 904.9998779296876,),
    (80.15997314453125, 904.3800659179688,),
    (79.88461303710938, 904.0768432617188,),
    (79.88461303710938, 904.0768432617188,),
    (79.4615707397461, 903.692138671875,),
    (79.23080444335938, 903.3460083007812,),
    (78.99999237060547, 903.0000610351562,),
    (78.74994659423828, 902.7501220703124,),
    (78.15385437011719, 902.2306518554688,),
    (77.40001678466797, 901.3001098632812,),
    (77.30000305175781, 901.10009765625,),
    (77.30000305175781, 901.10009765625,),
    (77.00000762939453, 900.5001220703125,),
    (76.80001068115234, 900.10009765625,),
    (76.50001525878906, 899.5001220703125,),
    (76.45001220703125, 899.3500366210938,),
    (75.8000259399414, 898.400146484375,),
    (76.0, 897.5,),
    (76.0, 897.5,),
    (75.6500015258789, 897.4500732421875,),
    (75.5, 897.0000610351562,),
    (75.38235473632812, 896.5294799804688,),
    (75.0, 896.5,),
    (75.0, 895.5,),
    (75.0, 894.5,),
    (75.0, 894.5,),
    (75.0, 894.5,),
    (74.5, 893.5,),
    (74.5, 893.0,),
    (74.5, 892.5,),
    (74.50000762939453, 892.0000610351562,),
    (74.5, 891.0,),
    (74.5, 890.5,),
    (74.5, 890.5,),
    (74.5, 890.5,),
    (74.5, 889.5,),
    (74.5, 889.0,),
    (74.0, 888.5,),
    (74.0, 887.5,),
    (74.5, 886.5,),
    (74.5, 886.0,),
    (74.5, 886.0,),
    (74.5, 885.5,),
    (74.5, 885.0,),
    (74.0, 884.5,),
    (74.0, 884.0,),
    (74.0, 882.5,),
    (74.0, 882.0,),
    (74.0, 882.0,),
    (74.0, 882.0,),
    (74.5, 881.0,),
    (74.5, 880.5,),
    (74.5, 879.5,),
    (74.5, 879.5,),
    (74.5, 878.0,),
    (74.5, 877.5,),
    (74.5, 877.5,),
    (74.5, 877.0,),
    (74.5, 876.5,),
    (74.5, 876.0,),
    (75.0, 875.5,),
    (75.0, 874.5,),
    (75.0, 874.0,),
    (75.0, 874.0,),
    (75.33782958984375, 873.4730224609375,),
    (75.3499984741211, 873.0498657226562,),
    (76.0, 872.5,),
    (76.0, 872.5,),
    (76.0, 871.5,),
    (76.0, 870.5,),
    (76.0, 870.0,),
    (76.0, 869.5,),
    (76.0, 869.5,),
    (76.0, 868.5,),
    (76.0, 868.5,),
    (76.0, 867.5,),
    (77.0, 867.5,),
    (77.0, 866.5,),
    (76.88236236572266, 866.470703125,),
    (77.0, 866.0,),
    (77.0, 866.0,),
    (77.0, 865.5,),
    (77.0, 865.5,),
    (77.0, 864.5,),
    (78.0, 864.0,),
    (78.0, 863.5,),
    (78.0, 863.0,),
    (78.0, 863.0,),
    (78.0, 863.0,),
    (78.0, 862.0,),
    (78.0, 862.0,),
    (79.0, 861.0,),
    (79.0, 860.5,),
    (79.0, 860.0,),
    (79.0, 860.0000610351562,),
    (79.0, 860.0000610351562,),
    (79.05883026123047, 859.7647705078125,),
    (79.2884521484375, 859.0577392578125,),
    (80.0, 859.0,),
    (80.0, 858.0,),
    (80.0, 858.0,),
    (80.0, 857.0,),
    (80.0, 856.5,),
    (80.0, 856.5,),
    (80.0, 856.0,),
    (80.0, 856.0,),
    (80.5, 855.0,),
    (80.5, 854.5,),
    (80.5, 854.0,),
    (80.5, 853.5,),
    (80.5, 853.5,),
    (80.5, 853.0,),
    (80.5, 852.5,),
    (81.24322509765625, 852.04052734375,),
    (81.5, 851.0,),
    (81.5, 850.5,),
    (81.5, 850.0,),
    (81.5, 849.0,),
    (81.5, 849.0,),
    (81.5, 848.5,),
    (81.5, 848.0,),
    (81.5, 846.5,),
    (81.0, 846.0000610351562,),
    (80.5, 845.5,),
    (80.5, 845.0,),
    (80.5, 845.0,),
    (80.5, 844.5,),
    (80.5, 844.0,),
    (80.5, 843.5,),
    (80.00000762939453, 842.0001220703125,),
    (80.0, 841.5,),
    (80.0, 840.5,),
    (80.0, 840.5,),
    (80.0, 840.5,),
    (80.0, 839.5,),
    (79.58000183105469, 838.5599975585938,),
    (79.5, 837.0,),
    (80.0, 836.5,),
    (80.0, 835.5,),
    (80.0, 835.5,),
    (80.0, 835.5,),
    (80.0, 834.5,),
    (80.0, 834.5,),
    (80.35295867919922, 833.5883178710938,),
    (80.0, 832.5,),
    (80.0, 832.0,),
    (80.0, 831.5,),
    (80.0, 830.5,),
    (80.0, 830.5,),
    (80.5, 830.0,),
    (80.5, 829.5,),
    (80.90383911132812, 828.4808349609375,),
    (81.07691192626953, 827.6154174804688,),
    (81.02940368652344, 827.3825073242188,),
    (81.15000915527344, 826.5501098632812,),
    (81.15000915527344, 826.5501098632812,),
    (81.34999084472656, 826.4500122070312,),
    (82.0, 825.5,),
    (81.99998474121094, 825.0000610351562,),
    (82.0, 824.0,),
    (82.4999771118164, 823.5000610351562,),
    (82.6500015258789, 823.050048828125,),
    (82.867919921875, 822.4622802734375,),
    (82.867919921875, 822.4622802734375,),
    (83.00001525878906, 822.0000610351562,),
    (83.26472473144531, 821.4412841796875,),
    (83.70000457763672, 820.6000366210938,),
    (84.0, 819.5,),
    (84.20000457763672, 819.4000244140625,),
    (84.30000305175781, 819.1000366210938,),
    (84.30000305175781, 819.1000366210938,),
    (84.0, 818.5,),
    (84.50000762939453, 818.5,),
    (84.80001068115234, 817.6000366210938,),
    (85.11763763427734, 816.529541015625,),
    (85.24999237060547, 816.2500610351562,),
    (85.65000915527344, 815.5501098632812,),
    (85.84998321533203, 814.9501342773438,),
    (85.79998016357422, 815.10009765625,),
    (85.94998168945312, 814.6500854492188,),
    (86.45001983642578, 814.1500244140625,),
    (86.80001068115234, 813.10009765625,),
    (86.76470947265625, 812.9412231445312,),
    (87.20589447021484, 812.1765747070312,),
    (87.26473236083984, 811.9412231445312,),
    (87.26473236083984, 811.9412231445312,),
    (87.40000915527344, 811.300048828125,),
    (87.94998168945312, 810.650146484375,),
    (88.05000305175781, 810.35009765625,),
    (88.79244995117188, 808.7264404296875,),
    (88.5, 808.5,),
    (88.5, 808.0,),
    (89.00000762939453, 807.4999389648438,),
    (89.00000762939453, 807.4999389648438,),
    (90.49999237060548, 801.0001831054688,),
    (90.48650360107422, 800.5810546875,),
    (90.56999969482422, 800.5099487304688,),
    (90.56999969482422, 800.5099487304688,),
    (90.50001525878906, 800.5000610351562,),
    (90.57693481445312, 800.1153564453125,),
    (90.59616088867188, 800.0191650390625,),
    (90.51924896240234, 799.9038696289062,),
    (90.70588684082031, 799.176513671875,),
    (90.72972106933594, 799.12158203125,),
    (90.5, 799.0,),
    (90.5, 799.0,),
    (91.0, 799.0,),
    (91.0, 799.0,),
    (90.5, 798.5,),
    (91.0, 799.0,),
    (90.79654693603516, 798.5587158203125,),
    (90.5, 799.0,),
    (90.5, 799.0,),
    (90.76470184326172, 805.9412841796875,),
    (91.0, 806.0,),
    (91.0, 806.5,),
    (90.69232940673828, 807.0385131835938,),
    (90.69232940673828, 807.0385131835938,),
    (90.5, 807.0,),
    (90.5, 807.5,),
    (90.5, 808.5,),
    (90.5, 808.5,),
    (90.5, 809.5,),
    (90.0, 809.5,),
    (90.0, 809.5,),
    (90.0, 810.0,),
    (90.0, 810.5,),
    (90.0, 810.5,),
    (90.0, 811.5,),
    (90.0, 811.5,),
    (90.0, 812.5,),
    (90.0, 812.5,),
    (90.0, 812.5,),
    (90.0, 813.5,),
    (90.0, 814.0,),
    (89.5, 815.0,),
    (89.5, 815.5,),
    (89.5, 816.0,),
    (89.5, 816.0,),
    (89.5, 816.0,),
    (89.5, 816.5,),
    (89.5, 817.0,),
    (89.5, 817.0,),
    (90.0, 817.5,),
    (89.5, 818.5,),
    (89.5, 818.5,),
    (90.0, 818.5,),
    (90.0, 818.5,),
    (90.0, 819.5,),
    (90.0, 820.0,),
    (90.0, 820.5,),
    (89.6500015258789, 820.9500732421875,),
    (89.8000259399414, 821.6000366210938,),
    (89.5, 822.0,),
    (89.5, 822.0,),
    (89.94998931884766, 822.3500366210938,),
    (89.91378021240234, 822.5345458984375,),
    (90.0862045288086, 822.965576171875,),
    (90.20000457763672, 823.400146484375,),
    (90.08621978759766, 823.965576171875,),
    (90.40003204345705, 824.3001098632812,),
    (90.00003814697266, 824.5001220703125,),
    (90.00003814697266, 824.5001220703125,),
    (90.20003509521484, 824.900146484375,),
    (90.60343170166016, 825.2586669921875,),
    (90.50003051757812, 825.5001220703125,),
    (91.00000762939452, 826.5000610351562,),
    (91.10000610351562, 826.7000732421875,),
    (91.2499771118164, 827.2501220703125,),
    (91.2499771118164, 827.2501220703125,),
    (91.50003814697266, 827.5001220703125,),
    (91.80001068115234, 828.1000366210938,),
    (92.20000457763672, 828.9000244140625,),
    (92.84617614746094, 829.7691040039062,),
    (93.00001525878906, 829.9998779296875,),
    (93.23077392578124, 830.3460693359375,),
    (93.5999984741211, 830.7000122070312,),
    (93.5999984741211, 830.7000122070312,),
    (93.80004119873048, 831.10009765625,),
    (93.89998626708984, 831.800048828125,),
    (94.38460540771484, 832.5768432617188,),
    (94.58001708984376, 832.9400634765625,),
    (95.00000762939452, 833.5001220703125,),
    (95.2307586669922, 833.845947265625,),
    (95.2307586669922, 833.845947265625,),
    (95.39998626708984, 834.300048828125,),
]

example_b = [
    # (x-Position, y-Position)
    (393.5, 370.5,),
    (393.5, 370.5,),
    (393.5, 370.5,),
    (393.5, 370.0,),
    (393.5, 370.0,),
    (393.50006103515625, 369.0,),
    (393.0, 369.0,),
    (393.0, 369.0,),
    (392.9000549316406, 369.1999816894531,),
    (392.9000549316406, 369.1999816894531,),
    (392.5000305175781, 369.0,),
    (392.6000671386719, 368.8000183105469,),
    (392.5000305175781, 369.0,),
    (392.2500305175781, 368.75,),
    (392.2500305175781, 368.75,),
    (391.5, 369.0,),
    (391.5, 369.0,),
    (391.5, 369.0,),
    (391.0500183105469, 369.1499938964844,),
    (390.9000854492188, 369.2000427246094,),
    (390.9000854492188, 369.2000427246094,),
    (390.9000854492188, 369.2000427246094,),
    (391.0, 369.5,),
    (390.5, 369.5,),
    (390.5, 369.5,),
    (390.5, 369.5,),
    (389.5, 369.5,),
    (389.5, 369.5,),
    (389.5, 369.5,),
    (389.0, 369.5,),
    (389.0, 369.5,),
    (389.0, 369.5,),
    (388.5, 369.5,),
    (388.0, 369.5,),
    (387.9117736816406, 369.6470642089844,),
    (387.60003662109375, 369.7000427246094,),
    (387.60003662109375, 369.7000427246094,),
    (387.6500244140625, 369.9499816894531,),
    (387.5000305175781, 369.9999694824219,),
    (387.5000305175781, 369.9999694824219,),
    (387.0500183105469, 370.1499633789063,),
    (386.5, 370.5,),
    (386.5, 370.5,),
    (386.5, 370.5,),
    (386.25, 370.7500305175781,),
    (385.6500244140625, 370.9500427246094,),
    (385.6500244140625, 370.9500427246094,),
    (385.5, 371.0000305175781,),
    (385.1538391113281, 371.23077392578125,),
    (385.2500305175781, 371.2500305175781,),
    (385.0000305175781, 371.5,),
    (384.75, 371.75006103515625,),
    (384.5000305175781, 372.0000305175781,),
    (384.5000305175781, 372.0000305175781,),
    (384.5000305175781, 372.0000305175781,),
    (384.5000305175781, 372.0000305175781,),
    (384.0000305175781, 372.5000305175781,),
    (384.0000305175781, 372.5000305175781,),
    (384.0000305175781, 372.5000305175781,),
    (384.25, 372.75,),
    (384.25, 372.75,),
    (384.25, 372.75,),
    (384.0, 373.0,),
    (383.9999694824219, 373.9999694824219,),
    (384.0, 374.0,),
    (384.0, 374.0,),
    (384.0, 374.0,),
    (384.3000183105469, 374.4000244140625,),
    (384.1000061035156, 374.800048828125,),
    (384.1000061035156, 374.800048828125,),
    (384.1000061035156, 374.800048828125,),
    (384.5, 375.5,),
    (384.5, 375.5,),
    (384.5, 375.5,),
    (384.5, 375.5,),
    (384.5, 375.5,),
    (384.5, 375.5,),
    (384.5, 375.5,),
    (385.5, 375.5,),
    (385.5, 375.5,),
    (385.5, 375.5,),
    (386.0, 375.5,),
    (385.8999938964844, 375.8000183105469,),
    (385.8999938964844, 375.8000183105469,),
    (386.0499877929688, 375.6500244140625,),
    (386.3000183105469, 375.6000061035156,),
    (386.3000183105469, 375.6000061035156,),
    (386.0, 375.5,),
    (386.0, 375.0,),
    (386.3000183105469, 374.60003662109375,),
    (386.0, 374.5,),
    (386.0, 374.5,),
    (386.1922912597656, 374.0384826660156,),
    (386.0, 373.5000305175781,),
    (385.95001220703125, 373.3500061035156,),
    (385.8000183105469, 372.9000244140625,),
    (385.6999816894531, 372.5999755859375,),
    (385.6500244140625, 372.4499816894531,),
    (385.5, 371.5,),
    (385.5, 371.5,),
    (385.5, 371.5,),
    (384.8823547363281, 370.5294189453125,),
    (384.8529357910156, 370.4117736816406,),
    (384.5, 369.5,),
    (384.5, 369.5,),
    (384.5, 369.5,),
    (384.5, 369.5,),
    (384.5, 369.5,),
    (384.0, 368.5,),
    (384.0, 368.5,),
    (384.0, 368.5,),
    (383.5, 368.5,),
    (383.5, 368.0,),
    (383.5, 368.0,),
    (383.5, 368.0,),
    (383.5, 367.5,),
    (383.5, 367.5,),
    (383.5, 366.5,),
    (383.5, 366.5,),
    (383.5, 366.0,),
    (383.5, 365.5,),
    (383.5, 365.5,),
    (383.5, 365.5,),
    (383.5, 365.0,),
    (383.5, 365.0,),
    (382.5, 365.0,),
    (382.5, 365.0,),
    (382.5, 364.5,),
    (382.5, 364.5,),
    (382.5, 364.5,),
    (382.5, 364.5,),
    (382.0, 364.0,),
    (382.0, 364.0,),
    (382.0, 364.0,),
    (382.0, 363.5,),
    (381.5, 363.5,),
    (381.5, 363.5,),
    (381.5, 363.5,),
    (381.2353515625, 362.94122314453125,),
    (381.0, 362.5,),
    (381.0, 362.5,),
    (380.5, 363.0,),
    (380.5, 363.0,),
    (380.5, 363.0,),
    (380.0, 363.0,),
    (380.4038391113281, 363.0192260742188,),
    (380.0, 363.5,),
    (380.0, 363.5,),
    (380.0, 363.5,),
    (380.0, 363.5,),
    (380.0, 364.0,),
    (380.0, 364.0,),
    (379.5, 364.5,),
    (379.5, 365.0,),
    (380.0, 365.5,),
    (380.0, 365.5,),
    (379.4117736816406, 365.85296630859375,),
    (379.2999877929688, 366.1000061035156,),
    (379.2999877929688, 366.1000061035156,),
    (379.0499877929688, 366.3500061035156,),
    (379.0, 366.5,),
    (379.0, 367.4999694824219,),
    (378.95001220703125, 367.6499938964844,),
    (379.0, 368.5000305175781,),
    (378.95001220703125, 368.6500549316406,),
    (378.70001220703125, 369.3999938964844,),
    (378.70001220703125, 369.3999938964844,),
    (378.70001220703125, 369.3999938964844,),
    (378.0, 370.5,),
    (378.5000305175781, 370.5000305175781,),
    (378.2999877929688, 371.1000061035156,),
    (378.2999877929688, 371.1000061035156,),
    (378.0, 371.5,),
    (378.0, 371.5,),
    (378.0000305175781, 372.0000305175781,),
    (378.0000305175781, 372.0000305175781,),
    (377.6999816894531, 372.8999938964844,),
    (377.3499755859375, 373.45001220703125,),
    (378.0, 373.5,),
    (377.31036376953125, 373.7241516113281,),
    (377.0500183105469, 374.3500061035156,),
    (377.0500183105469, 374.3500061035156,),
    (377.0, 374.4999694824219,),
    (376.70001220703125, 375.60003662109375,),
    (376.0, 376.0,),
    (376.0, 376.5,),
    (376.0000305175781, 377.0000305175781,),
    (376.0, 377.5,),
    (376.0, 377.5,),
    (376.0, 378.0,),
    (376.0, 378.5,),
    (376.0, 379.5,),
    (375.3999938964844, 379.8000183105469,),
    (375.5, 380.0,),
    (375.5, 381.0,),
    (375.5, 381.0,),
    (375.5, 381.0,),
    (375.5, 381.5,),
    (375.1153869628906, 382.423095703125,),
    (374.9118041992188, 382.8529968261719,),
    (374.8529968261719, 383.08831787109375,),
    (374.7353515625, 383.5588684082031,),
    (374.6470947265625, 383.9118347167969,),
    (374.6470947265625, 383.9118347167969,),
    (374.5, 384.0,),
    (374.5, 384.0,),
    (374.5, 385.0,),
    (374.5, 385.5,),
    (374.5, 386.0,),
    (374.5, 386.0,),
    (374.5, 386.5,),
    (374.5, 387.0,),
    (374.5, 387.0,),
    (374.5, 387.5,),
    (374.5, 387.5,),
    (374.5, 388.0,),
    (374.5, 388.0,),
    (374.5, 388.5,),
    (374.5, 389.0,),
    (374.5, 389.0,),
    (374.5, 389.5,),
    (374.5, 389.5,),
    (374.5, 390.0,),
    (374.5, 390.0,),
    (374.5, 390.5,),
    (374.5, 391.0,),
    (374.5, 391.0,),
    (374.5, 391.0,),
    (374.5, 391.5,),
    (374.5, 391.5,),
    (374.5, 391.5,),
    (374.5, 392.0,),
    (374.5, 392.0,),
    (374.5, 392.5,),
    (374.5, 392.5,),
    (374.5, 392.5,),
    (375.0000305175781, 392.50006103515625,),
    (374.8823852539063, 392.5294799804688,),
    (375.0000305175781, 392.50006103515625,),
    (375.0294494628906, 392.6177062988281,),
    (375.0, 392.5,),
    (375.2999877929688, 392.9000244140625,),
    (375.2999877929688, 392.9000244140625,),
    (375.4999694824219, 393.0000305175781,),
    (375.6499633789063, 393.4500427246094,),
    (375.6999816894531, 393.4000549316406,),
    (375.9000244140625, 393.300048828125,),
    (375.8000183105469, 393.10003662109375,),
    (375.8999938964844, 392.800048828125,),
    (376.0, 392.5,),
    (376.0, 392.5,),
    (376.0, 392.0,),
    (376.0, 392.0,),
    (376.25, 391.7500305175781,),
    (375.9000244140625, 391.3000793457031,),
    (375.8000183105469, 391.1000671386719,),
    (376.0000305175781, 391.00006103515625,),
    (376.0000305175781, 391.00006103515625,),
    (375.7500305175781, 390.25006103515625,),
    (375.75, 390.2500915527344,),
    (375.5000305175781, 390.00006103515625,),
    (375.4615173339844, 389.6922607421875,),
    (375.25, 389.2500305175781,),
    (374.9999694824219, 388.9999694824219,),
    (374.7692260742188, 388.65380859375,),
    (374.7692260742188, 388.65380859375,),
    (374.3000183105469, 388.10003662109375,),
    (374.0000305175781, 387.50006103515625,),
    (373.6999816894531, 387.4000549316406,),
    (374.0, 387.0,),
    (373.3500061035156, 386.5500183105469,),
    (373.2000427246094, 386.4000854492188,),
    (373.2000427246094, 386.4000854492188,),
    (372.70001220703125, 385.6000061035156,),
    (372.5, 385.5,),
    (372.3823547363281, 385.0293884277344,),
    (372.0, 384.5,),
    (371.5, 384.5,),
    (371.5, 384.5,),
    (371.5, 384.0,),
    (371.5, 384.0,),
    (371.5, 384.0,),
    (370.5, 384.0,),
    (370.5, 383.5,),
    (370.5, 383.5,),
    (370.5, 383.5,),
    (370.5, 383.0,),
    (370.5, 383.0,),
    (370.5, 382.5,),
    (370.5, 382.0,),
    (370.5, 382.0,),
    (370.5, 381.5,),
    (370.0, 381.5,),
    (370.0, 381.5,),
    (370.0, 381.0,),
    (370.0, 381.0,),
    (370.0, 381.5,),
    (369.4000244140625, 381.2000732421875,),
    (369.20001220703125, 381.6000671386719,),
    (369.20001220703125, 381.6000671386719,),
]

if __name__ == '__main__':
    from gsff.main import GaussianSumFIR

    gsff = GaussianSumFIR(delta_t=.1)

    try:
        from matplotlib import pyplot as plt

        for example in [example_a[:], example_b[:]]:

            settings_dict = {}
            gsff_result = []
            gsff_pred = []
            for mes in example:
                result, settings_dict = gsff.correct(mes, **settings_dict)
                gsff_result.append(result)

                pred, settings_dict = gsff.predict(**settings_dict)
                gsff_pred.append(pred)
            fig, ax = plt.subplots()
            for curr_plot in [example, gsff_pred[:], gsff_result[:]]:
                ax.plot(
                    [i[0] for i in curr_plot],
                    [i[1] for i in curr_plot],
                    # marker='o',
                    # color='black',
                    # s=1,
                    # lw=0,
                )
                ax.scatter(curr_plot[0][0], curr_plot[0][1], marker='+', color='r')
            fig.show()
    except ImportError:
        print('matplotlib could not be imported.')