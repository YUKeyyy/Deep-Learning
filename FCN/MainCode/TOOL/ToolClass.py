import math
import matplotlib.pyplot as plt
import numpy as np
from pylab import *
from scipy import fft
from scipy.fftpack import fft
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import MinMaxScaler

def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma

def normalization(data):
    _range = np.max(abs(data))
    return data / _range

# region 地震图绘制
def __Draw__(A):
    figure, ax = plt.subplots()
    # region 设置x，y值域
    ax = plt.gca()
    ax.xaxis.set_ticks_position('top')
    ax.invert_yaxis()
    # endregion

    for i in range(len(A)):
        A[i,:] = normalization(A[i,:])

    for j in range(0, A.shape[0]):
        Y_value = A[j] + 2 * j
        X = 2 * j
        i = 0
        X_value = [i for i in range(1, A.shape[1] + 1)]
        plt.plot((X, X), (0, 1500), linewidth='0.5', color='black')
        plt.plot(Y_value, X_value, linewidth='0.5', color='black')
        plt.fill_betweenx(X_value, Y_value, X, where=Y_value > X, facecolor='black', interpolate=True)
    plt.xlabel('Y')
    plt.ylabel('X')
    # plt.show()


# endregion

# region加载数据
def __LoadData__(name, row, col):
    print('Loading...Data...')
    A = np.zeros((row, col))
    A_row = 0
    src = './DataResource/'
    src = src + name
    fo = open(src, 'r')  # 打开数据文件文件
    lines = fo.readlines()

    for line in lines:  # 把lines中的数据逐行读取出来
        if A_row < row:
            List = line.strip().split()  # 处理逐行数据：strip表示把头尾的'\n'去掉，split表示以空格来分割行数据，然后把处理后的行数据返回到list列表中
            A[A_row:] = List[0:col]  # 把处理后的数据放到方阵A中。list[0:3]表示列表的0,1,2列数据放到矩阵A中的A_row行
            A_row += 1  # 然后方阵A的下一行接着读
        else:
            break
    fo.close()
    print('End...__LoadData__')
    return A


# endregion

# region 单道SVD
def __SVD__(A, beginTime, endTime):
    pr = PrintTool()
    pr.isShow = False
    # pr.__print__('SVD...loading...')
    # pr.__print__('beginTime= '+str(beginTime)+'...endTime= '+str(endTime))
    # pr.__print__('The type of A is :')
    # pr.__print__(str(type(A)))
    # pr.__print__('The A is :')
    # pr.__print__(str(A))
    U, sigma, VT = np.linalg.svd(A)
    sigma_len = len(sigma)
    # print('sigma: '+str(len(sigma)))
    # 处理二维矩阵
    shape = np.shape(A)
    row = shape[0]
    col = shape[1]
    dig_len = len(sigma)
    sigma_len = dig_len
    Derta = sigma
    B = np.zeros((1, sigma_len - 1))
    B = B[0]
    Max_Value = 0
    Max_Second_Value = 0
    Max_Index = 0
    Max_Second_Index = 0
    pr.__print__('A:shape')
    pr.__print__(str(np.shape(A)))

    # '奇异值差分谱'
    for i in range(sigma_len - 1):
        B[i] = Derta[i] - Derta[i + 1]
        # print(B[i])
        if B[i] > Max_Value:
            Max_Value = B[i]
            Max_Index = i
        elif B[i] > Max_Second_Value:
            Max_Second_Value = B[i]
            Max_Second_Index = i

    # if Max_Index == 0:
    #      Max_Index = Max_Second_Index
    Max_Index += 1
    Max_Second_Index +=1
    # print(Max_Index)
    # dig = (np.mat(row, int(np.ceil(dig_len))) * sigma)  #np.eye(int(np.ceil(dig_len)), int(np.ceil(dig_len))) * sigma
    redata = (U[:, 0:Max_Index]).dot(np.diag(sigma[0:Max_Index])).dot(VT[0:Max_Index, :])


    # dig = np.mat(np.eye(int(np.ceil(row * (endTime)) - np.ceil(row * beginTime) + 1)) * sigma[int(np.ceil(row * beginTime)) - 1:int(np.ceil(row * endTime))])
    dig = np.mat(np.eye(row, int(np.ceil(dig_len))) * sigma)

    pr.__print__('dig is :')
    pr.__print__(str(np.shape(dig)))
    # 获得对角矩阵
    redata = U[:, int(np.ceil(dig_len * beginTime)):int(np.ceil(dig_len * endTime))] * dig[int(
        np.ceil(dig_len * beginTime)):int(np.ceil(dig_len * endTime)), int(np.ceil(dig_len * beginTime)):int(
        np.ceil(dig_len * endTime))] * VT[int(np.ceil(dig_len * beginTime)):int(np.ceil(dig_len * endTime)), :]
    # redata = U * dig * VT
    Rebuilddata_cover = np.array(redata)
    # pr.__print__('Rebuilddata_cover is :')
    # pr.__print__(str(Rebuilddata_cover))
    # pr.__print__('SVD...ending...')
    return Rebuilddata_cover


# endregion


# region Hankel矩阵转换
def __SingalToHankel__(A):
    pr = PrintTool()
    pr.isShow = False
    n = len(A)
    string = 'The size of single A is :' + str(n)
    pr.__print__(string)
    pr.__print__('A type :' + str(type(A)))
    pr.__print__('A is :')
    pr.__print__(str(A))

    m = int(n / 2) + 1
    A = np.array(A)
    H = np.zeros((m, (n - m + 1)))

    for i in range(m):
        H[i] = A[i:n - m + i + 1]
    # H = scipy.linalg.hankel(A[:n-m+1],A[m:])
    pr.__print__('H type :' + str(type(H)))
    pr.__print__('H is :' + str(np.shape(H)))
    string = 'The hankel H is :\n' + str(H)
    pr.__print__(string)
    return H


# endregion

# hankel--->single
# region
def __HankeltoSingle__(A):
    m = A.shape[0]
    n = m + A.shape[1] - 1
    H = np.zeros((1, n))
    for i in range(A.shape[0]):
        H[0, i:n - m + i + 1] = A[i]
    return H[0]


# endregion
# region 频域图像
def __DrawFrequencyDomain__(A, x):
    col = len(A)
    # 设置需要采样的信号，频率分量有200，400和600
    # y = 7 * np.sin(2 * np.pi * 200 * x) + 5 * np.sin(2 * np.pi * 400 * x) + 3 * np.sin(2 * np.pi * 600 * x)

    plt.figure()
    plt.plot(x, A)
    plt.title('原始波形')
    # plt.show()

    # 快速傅里叶变换
    fft_A = fft(A)
    print(fft_A)
    '变换之后的结果数据长度和原始采样信号是一样的'
    '每一个变换之后的值是一个复数，为a+bj的形式'
    '复数a+bj在坐标系中表示为（a,b），故而复数具有模和角度'
    '快速傅里叶变换具有 “振幅谱”“相位谱”，它其实就是通过对快速傅里叶变换得到的复数结果进一步求出来的'
    '那这个直接变换后的结果是需要的，在FFT中，得到的结果是复数'
    'FFT得到的复数的模（即绝对值）就是对应的“振幅谱”，复数所对应的角度，就是所对应的“相位谱”'

    # FFT的原始频谱
    # 取复数的绝对值，即复数的模(双边频谱)
    abs_A = np.abs(fft_A)
    # 取复数的角度
    angle_A = np.angle(fft_A)

    plt.figure()
    plt.plot(x, abs_A)
    plt.title('双边振幅谱（未归一化）')

    plt.figure()
    plt.plot(x, angle_A)
    plt.title('双边相位谱（未归一化）')

    '我们在此处仅仅考虑“振幅谱”，不再考虑相位谱。'
    '我们发现，振幅谱的纵坐标很大，而且具有对称性，这是怎么一回事呢？'
    '关于振幅值很大的解释以及解决办法——归一化和取一半处理'
    '''
    比如有一个信号如下：
    Y=A1+A2*cos(2πω2+φ2）+A3*cos(2πω3+φ3）+A4*cos(2πω4+φ4）
    经过FFT之后，得到的“振幅图”中，
    第一个峰值（频率位置）的模是A1的N倍，N为采样点，本例中为N=1400，此例中没有，因为信号没有常数项A1
    第二个峰值（频率位置）的模是A2的N/2倍，N为采样点，
    第三个峰值（频率位置）的模是A3的N/2倍，N为采样点，
    第四个峰值（频率位置）的模是A4的N/2倍，N为采样点，
    依次下去......
    考虑到数量级较大，一般进行归一化处理，既然第一个峰值是A1的N倍，那么将每一个振幅值都除以N即可
    FFT具有对称性，一般只需要用N的一半，前半部分即可。
    '''

    # 归一化
    normalization_A = abs_A / col
    plt.figure()
    plt.plot(x, normalization_A, 'g')
    plt.title('双边频谱(归一化)', fontsize=9, color='green')

    # 取半处理
    half_x = x[range(int(col / 2))]  # 取一半区间
    normalization_half_A = normalization_A[range(int(col / 2))]  # 由于对称性，只取一半区间（单边频谱）
    plt.figure()
    plt.plot(half_x[0:50], normalization_half_A[0:50], 'b')
    plt.title('单边频谱(归一化)', fontsize=9, color='blue')


# endregion


class PrintTool(object):
    isShow = True

    def __print__(self, str):
        strl = '\n' + str
        if self.isShow == True:
            print(strl)
