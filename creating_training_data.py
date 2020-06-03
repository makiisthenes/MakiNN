# Michael Peres
import numpy as np

# Generating Random Data (just for understanding not for practicality...)
# This is quite complicated not sure what this code does, just copied and pasted.. but not priority.
# np.random.seed(0)

def create_data(points, classes):
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points)  # radius
        t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.05
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X, y

import matplotlib.pyplot as plt
X, y = create_data(100, 3)
# print(X)
# print(y)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()
for x in range(len(X)):
    print(f'{X[x]} -- {y[x]}')

'''
[0. 0.] -- 0
[0.00059984 0.01008318] -- 0
[0.00831931 0.01840953] -- 0
[0.00858094 0.02906271] -- 0
[0.01440768 0.03774792] -- 0
[0.02405828 0.04440675] -- 0
[0.04152148 0.04414818] -- 0
[0.04458051 0.05488231] -- 0
[0.04000558 0.0702104 ] -- 0
[0.06837601 0.0599098 ] -- 0
[0.09336251 0.03855494] -- 0
[0.10265169 0.04252421] -- 0
[0.11873099 0.02439939] -- 0
[0.12627702 0.03601741] -- 0
[0.14034479 0.01735799] -- 0
[ 0.15148511 -0.00301707] -- 0
[ 0.16076316 -0.01658278] -- 0
[ 0.1713959  -0.01049918] -- 0
[ 0.17292793 -0.05615853] -- 0
[ 0.18482976 -0.0516811 ] -- 0
[ 0.19005598 -0.06849003] -- 0
[ 0.17767867 -0.11586931] -- 0
[ 0.14109045 -0.17168634] -- 0
[ 0.1593862 -0.169027 ] -- 0
[ 0.15780456 -0.18403052] -- 0
[ 0.10389957 -0.23016056] -- 0
[ 0.0915852 -0.2461396] -- 0
[ 0.10115678 -0.25327351] -- 0
[ 0.06946147 -0.2741659 ] -- 0
[ 0.04578729 -0.2893287 ] -- 0
[ 0.02374432 -0.30209861] -- 0
[ 0.00802647 -0.31302843] -- 0
[-0.07414436 -0.31461365] -- 0
[-0.14764778 -0.29884987] -- 0
[-0.16331053 -0.30212054] -- 0
[-0.10927814 -0.33622245] -- 0
[-0.2033559  -0.30145942] -- 0
[-0.16053895 -0.33750092] -- 0
[-0.27789722 -0.26477356] -- 0
[-0.28660376 -0.27027122] -- 0
[-0.26678683 -0.30343605] -- 0
[-0.32048812 -0.26229845] -- 0
[-0.37487076 -0.19862917] -- 0
[-0.37508251 -0.21901446] -- 0
[-0.44355176 -0.02815493] -- 0
[-0.45191088 -0.0488685 ] -- 0
[-0.46399387  0.02461759] -- 0
[-0.46314033  0.10433693] -- 0
[-0.48223776 -0.05024737] -- 0
[-0.474842    0.13964269] -- 0
[-0.48057023  0.15533275] -- 0
[-0.45714197  0.23749169] -- 0
[-0.40958132  0.32883637] -- 0
[-0.42588903  0.32437932] -- 0
[-0.39923244  0.37166399] -- 0
[-0.36260189  0.42090598] -- 0
[-0.322394    0.46478969] -- 0
[-0.27085596  0.50806873] -- 0
[-0.27221041  0.51877912] -- 0
[-0.12425473  0.58286242] -- 0
[-0.09904841  0.59791209] -- 0
[-0.1198564   0.60439191] -- 0
[0.04723476 0.62447879] -- 0
[-0.100101   0.6284413] -- 0
[0.24945855 0.59639498] -- 0
[0.10055803 0.64881935] -- 0
[0.32476426 0.58221355] -- 0
[0.33540512 0.5878077 ] -- 0
[0.2488504  0.64020471] -- 0
[0.55945051 0.4156704 ] -- 0
[0.56079654 0.43064629] -- 0
[0.48356481 0.52962283] -- 0
[0.5186613  0.50981965] -- 0
[0.62536327 0.39069273] -- 0
[0.70364962 0.25218191] -- 0
[0.7575024  0.01054221] -- 0
[0.76169438 0.0956519 ] -- 0
[0.76863301 0.11891829] -- 0
[0.78710155 0.03498756] -- 0
[ 0.79593873 -0.05703762] -- 0
[ 0.80269519 -0.09313984] -- 0
[ 0.78160553 -0.24189727] -- 0
[ 0.73928013 -0.37352019] -- 0
[ 0.75675936 -0.36083615] -- 0
[ 0.65080455 -0.54440792] -- 0
[ 0.5519102  -0.65769659] -- 0
[ 0.60244145 -0.62584437] -- 0
[ 0.54691873 -0.68785757] -- 0
[ 0.54753119 -0.70023786] -- 0
[ 0.28699624 -0.85194835] -- 0
[ 0.37882565 -0.82640027] -- 0
[ 0.32525277 -0.85972346] -- 0
[ 0.18526511 -0.91063834] -- 0
[ 0.02711948 -0.9390024 ] -- 0
[-0.00812684 -0.94946017] -- 0
[ 0.0712022  -0.95695071] -- 0
[-0.34211778 -0.90734097] -- 0
[-0.40425153 -0.89251598] -- 0
[-0.4983319  -0.85531592] -- 0
[-0.47223155 -0.88147454] -- 0
[-0. -0.] -- 1
[-0.00758775 -0.00666756] -- 1
[-0.01726457 -0.01049076] -- 1
[-0.02381847 -0.01873377] -- 1
[-0.03314751 -0.02310257] -- 1
[-0.04330549 -0.02598836] -- 1
[-0.05437109 -0.02677461] -- 1
[-0.06090823 -0.03591207] -- 1
[-0.07984402  0.012445  ] -- 1
[-0.09058367 -0.0076852 ] -- 1
[-0.10027741 -0.0121442 ] -- 1
[-0.11085555  0.00753163] -- 1
[-0.1141364  0.0408076] -- 1
[-0.12756332  0.03115666] -- 1
[-0.12505719  0.06602013] -- 1
[-0.10760321  0.10666954] -- 1
[-0.13937816  0.08181389] -- 1
[-0.15492827  0.07405416] -- 1
[-0.09768017  0.1533507 ] -- 1
[-0.12139841  0.14864522] -- 1
[-0.1316236   0.15325596] -- 1
[-0.11135941  0.18053945] -- 1
[-0.07828233  0.20797739] -- 1
[-0.09107397  0.2137279 ] -- 1
[-0.00135085  0.24242048] -- 1
[-0.01550904  0.25204855] -- 1
[0.00042561 0.26262592] -- 1
[0.04184587 0.26949785] -- 1
[0.14913341 0.24031451] -- 1
[0.09968899 0.2754445 ] -- 1
[0.10139762 0.28556241] -- 1
[0.08030083 0.30265987] -- 1
[0.16799934 0.27614372] -- 1
[0.22600581 0.24501528] -- 1
[0.29963958 0.16781915] -- 1
[0.29388328 0.19651938] -- 1
[0.34286627 0.1211368 ] -- 1
[0.32385902 0.18653408] -- 1
[0.37207368 0.09430314] -- 1
[0.38814343 0.06732702] -- 1
[ 0.40022023 -0.05542935] -- 1
[0.41305044 0.03004073] -- 1
[0.42292031 0.03346706] -- 1
[ 0.41370133 -0.13230809] -- 1
[ 0.41943956 -0.14697389] -- 1
[ 0.44346988 -0.09972982] -- 1
[ 0.43935609 -0.15120373] -- 1
[ 0.36331    -0.30559943] -- 1
[ 0.32335794 -0.36127233] -- 1
[ 0.28289129 -0.40613732] -- 1
[ 0.35398605 -0.36023588] -- 1
[ 0.28194658 -0.43114639] -- 1
[ 0.14872892 -0.50375582] -- 1
[ 0.15550201 -0.51227193] -- 1
[ 0.18618707 -0.5126939 ] -- 1
[ 0.05652175 -0.55267284] -- 1
[ 0.05063411 -0.56338578] -- 1
[-0.04645139 -0.5738807 ] -- 1
[-0.16042361 -0.56346654] -- 1
[-0.13860273 -0.57961808] -- 1
[-0.17586991 -0.5799821 ] -- 1
[-0.34071426 -0.51338965] -- 1
[-0.31699833 -0.54010826] -- 1
[-0.2784793  -0.57219573] -- 1
[-0.34006379 -0.54979374] -- 1
[-0.54626978 -0.36423589] -- 1
[-0.52298759 -0.41343491] -- 1
[-0.58829852 -0.33454348] -- 1
[-0.59323618 -0.3462072 ] -- 1
[-0.67633664 -0.16833153] -- 1
[-0.6862169  -0.17045632] -- 1
[-0.70943775 -0.10503975] -- 1
[-0.72726651  0.00300621] -- 1
[-0.72457154  0.13680682] -- 1
[-0.73404828  0.14103767] -- 1
[-0.73468004  0.18484119] -- 1
[-0.60532756  0.47212939] -- 1
[-0.65876371  0.41348354] -- 1
[-0.55384697  0.56036285] -- 1
[-0.4534927   0.65659434] -- 1
[-0.67343182  0.44663651] -- 1
[-0.51165377  0.63846058] -- 1
[-0.52172416  0.64331667] -- 1
[-0.44456451  0.7108093 ] -- 1
[-0.39086094  0.75309645] -- 1
[-0.20678765  0.83331179] -- 1
[-0.09081796  0.86392649] -- 1
[-0.09685411  0.87343427] -- 1
[0.2001482  0.86606244] -- 1
[0.10959703 0.89228433] -- 1
[0.13786813 0.89857591] -- 1
[0.48318983 0.78194717] -- 1
[0.53415223 0.76043852] -- 1
[0.45039362 0.82438253] -- 1
[0.60400124 0.73261393] -- 1
[0.6144248  0.73709332] -- 1
[0.67696733 0.69428196] -- 1
[0.86762145 0.45523301] -- 1
[0.92520067 0.35199962] -- 1
[0.85395769 0.52034245] -- 1
[0. 0.] -- 2
[0.00936693 0.00378036] -- 2
[0.01878707 0.00742748] -- 2
[0.02994492 0.00464494] -- 2
[ 0.03955221 -0.00825284] -- 2
[ 0.05034753 -0.00398571] -- 2
[ 0.06057982 -0.00178311] -- 2
[ 0.06415465 -0.02972659] -- 2
[ 0.07230739 -0.03607752] -- 2
[ 0.08003252 -0.04311913] -- 2
[ 0.07107471 -0.07177344] -- 2
[ 0.07837065 -0.0787637 ] -- 2
[ 0.09645288 -0.07341131] -- 2
[ 0.08902514 -0.09652805] -- 2
[ 0.06842879 -0.12375565] -- 2
[ 0.05056465 -0.14282877] -- 2
[ 0.04858816 -0.15413946] -- 2
[ 0.08307587 -0.15028369] -- 2
[ 0.04385035 -0.17645112] -- 2
[ 0.00706256 -0.1917892 ] -- 2
[-0.04728839 -0.19640766] -- 2
[-0.02050201 -0.2111281 ] -- 2
[-0.06212258 -0.21336237] -- 2
[-0.0243869  -0.23103974] -- 2
[-0.09879788 -0.22137862] -- 2
[-0.1811156  -0.17597199] -- 2
[-0.18953035 -0.18179879] -- 2
[-0.17552653 -0.20873572] -- 2
[-0.18561227 -0.21340085] -- 2
[-0.26035407 -0.134251  ] -- 2
[-0.27119978 -0.13519633] -- 2
[-0.30472076 -0.0720866 ] -- 2
[-0.28568565 -0.15120464] -- 2
[-0.33321684 -0.00881204] -- 2
[-0.34342557  0.00245451] -- 2
[-0.34470624 -0.07851659] -- 2
[-0.36309508  0.01983354] -- 2
[-0.37132917  0.04235887] -- 2
[-0.35077124  0.15585713] -- 2
[-0.36930877  0.13711045] -- 2
[-0.3458083   0.20896236] -- 2
[-0.33943035  0.23727652] -- 2
[-0.35719029  0.22890333] -- 2
[-0.2403674   0.36177027] -- 2
[-0.35646111  0.26545497] -- 2
[-0.2017561   0.40731566] -- 2
[-0.10984664  0.45147542] -- 2
[-0.1488341   0.45081435] -- 2
[-0.15798671  0.45838658] -- 2
[-0.14910911  0.47195495] -- 2
[0.01963149 0.50466882] -- 2
[0.08533768 0.50803402] -- 2
[-0.00560856  0.52522258] -- 2
[0.20467879 0.49468172] -- 2
[0.18449286 0.513306  ] -- 2
[0.24548964 0.49837417] -- 2
[0.33916321 0.45269821] -- 2
[0.36214498 0.44760228] -- 2
[0.35294026 0.46761464] -- 2
[0.45410774 0.3859456 ] -- 2
[0.49995738 0.3425669 ] -- 2
[0.54549086 0.286522  ] -- 2
[0.50155626 0.37502826] -- 2
[0.63235656 0.0713012 ] -- 2
[0.64501526 0.04326486] -- 2
[ 0.65517094 -0.04277262] -- 2
[0.66342785 0.06563489] -- 2
[ 0.66536081 -0.1237315 ] -- 2
[ 0.68503351 -0.05017655] -- 2
[ 0.69445458 -0.05915735] -- 2
[ 0.6694673  -0.22751377] -- 2
[ 0.61969968 -0.36098142] -- 2
[ 0.57178213 -0.44943388] -- 2
[ 0.58553951 -0.44817799] -- 2
[ 0.61760253 -0.42105298] -- 2
[ 0.38862032 -0.65030399] -- 2
[ 0.37281594 -0.67107071] -- 2
[ 0.43626853 -0.64390065] -- 2
[ 0.3657189  -0.69785577] -- 2
[ 0.2355832  -0.76241217] -- 2
[ 0.03613249 -0.80727259] -- 2
[-0.15811904 -0.80275766] -- 2
[-0.13675394 -0.81691542] -- 2
[ 0.00457053 -0.83837138] -- 2
[-0.09312705 -0.8433587 ] -- 2
[-0.21756275 -0.83056374] -- 2
[-0.47861682 -0.72494332] -- 2
[-0.42664588 -0.76827172] -- 2
[-0.47557729 -0.75096584] -- 2
[-0.66901034 -0.60050646] -- 2
[-0.71316745 -0.56377164] -- 2
[-0.56083435 -0.72827097] -- 2
[-0.79100212 -0.48775095] -- 2
[-0.83024996 -0.43948376] -- 2
[-0.94572006 -0.08458263] -- 2
[-0.8346354  -0.47350623] -- 2
[-0.96372777 -0.10742908] -- 2
[-0.9759892   0.08630851] -- 2
[-0.98171276  0.12704358] -- 2
[-0.99958662  0.02875052] -- 2
'''
