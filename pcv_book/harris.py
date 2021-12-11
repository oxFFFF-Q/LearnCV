from pylab import *
from numpy import *
from scipy.ndimage import filters


def compute_harris_response(im,sigma=3):
    """ Compute the Harris corner detector response function 
        for each pixel in a graylevel image. """
    # 在灰度图中，对每个像素计算Harris角点检测器响应函数
    
    # derivatives 计算导数
    imx = zeros(im.shape)
    filters.gaussian_filter(im, (sigma,sigma), (0,1), imx)
    imy = zeros(im.shape)
    filters.gaussian_filter(im, (sigma,sigma), (1,0), imy)
    
    # compute components of the Harris matrix
    Wxx = filters.gaussian_filter(imx*imx,sigma)
    Wxy = filters.gaussian_filter(imx*imy,sigma)
    Wyy = filters.gaussian_filter(imy*imy,sigma)
    
    # determinant and trace 计算特征值和轨迹
    Wdet = Wxx*Wyy - Wxy**2
    Wtr = Wxx + Wyy
    
    return Wdet / (Wtr*Wtr)
   
    
def get_harris_points(harrisim,min_dist=10,threshold=0.1):
    """ Return corners from a Harris response image
        min_dist is the minimum number of pixels separating 
        corners and image boundary. """
    """ 从 Harris 响应图像返回角点。 min_dist 是分隔角点和图像边界的最小像素数。 """
    
    # find top corner candidates above a threshold
    # 找到高于闸值的候选角点
    corner_threshold = harrisim.max() * threshold
    harrisim_t = (harrisim > corner_threshold) * 1
    
    # get coordinates of candidates
    # 得到候选角点坐标
    coords = array(harrisim_t.nonzero()).T
    
    # ...and their values
    # 以及它们的Harris响应值
    candidate_values = [harrisim[c[0],c[1]] for c in coords]
    
    # sort candidates
    # 对候选点按照响应值进行排序
    index = argsort(candidate_values)
    
    # store allowed point locations in array
    # 将可行点的位置保存在数组中
    allowed_locations = zeros(harrisim.shape)
    allowed_locations[min_dist:-min_dist,min_dist:-min_dist] = 1
    
    # select the best points taking min_distance into account
    # 按照min_distance原则，选取最佳的Harris点
    filtered_coords = []
    for i in index:
        if allowed_locations[coords[i,0],coords[i,1]] == 1:
            filtered_coords.append(coords[i])
            allowed_locations[(coords[i,0]-min_dist):(coords[i,0]+min_dist), 
                        (coords[i,1]-min_dist):(coords[i,1]+min_dist)] = 0
    
    return filtered_coords
    
    
def plot_harris_points(image,filtered_coords):
    """ Plots corners found in image. """
    """ 绘制检测到的角点 """
    
    figure()
    gray()
    imshow(image)
    plot([p[1] for p in filtered_coords],
                [p[0] for p in filtered_coords],'*')
    axis('off')
    show()
    '''
    im = array(Image.open('xxx.jpg').convert('L'))
    harrisim = harris.compute_harris_response(im)
    filtered_coords = harris.get_harris_points(harrisim,6)
    harris.plot_harris_points(im,filtered_coords)
    '''
    

def get_descriptors(image,filtered_coords,wid=5):
    """ For each point return pixel values around the point
        using a neighbourhood of width 2*wid+1. (Assume points are 
        extracted with min_distance > wid). """
    '''
    对于每个返回的点，返回点周围2*wid+1个像素的值（假设选取点的min_distance > wid）
    '''
    
    desc = []
    for coords in filtered_coords:
        patch = image[coords[0]-wid:coords[0]+wid+1,
                            coords[1]-wid:coords[1]+wid+1].flatten()
        desc.append(patch)
    
    return desc


def match(desc1,desc2,threshold=0.5):
    """ For each corner point descriptor in the first image, 
        select its match to second image using
        normalized cross correlation. """
    '''
    对于第一幅图像中的每个角点描述子，使用归一化互相关，选取它在第二幅图像中的匹配角点
    '''
    n = len(desc1[0])
    
    # pair-wise distances
    # 点对的距离
    d = -ones((len(desc1),len(desc2)))
    for i in range(len(desc1)):
        for j in range(len(desc2)):
            d1 = (desc1[i] - mean(desc1[i])) / std(desc1[i])
            d2 = (desc2[j] - mean(desc2[j])) / std(desc2[j])
            ncc_value = sum(d1 * d2) / (n-1) 
            if ncc_value > threshold:
                d[i,j] = ncc_value
            
    ndx = argsort(-d)
    matchscores = ndx[:,0]
    
    return matchscores


def match_twosided(desc1,desc2,threshold=0.5):
    """ Two-sided symmetric version of match(). """
    ''' 两边对称版本的match '''
    
    matches_12 = match(desc1,desc2,threshold)
    matches_21 = match(desc2,desc1,threshold)
    
    ndx_12 = where(matches_12 >= 0)[0]
    
    # remove matches that are not symmetric
    # 去除非对称的匹配
    for n in ndx_12:
        if matches_21[matches_12[n]] != n:
            matches_12[n] = -1
    
    return matches_12


def appendimages(im1,im2):
    """ Return a new image that appends the two images side-by-side. """
    ''' 返回两幅图像拼接成的新图像'''
    
    # select the image with the fewest rows and fill in enough empty rows
    # 选取具有最少行数的图像，然后填充足够的空行
    rows1 = im1.shape[0]    
    rows2 = im2.shape[0]
    
    if rows1 < rows2:
        im1 = concatenate((im1,zeros((rows2-rows1,im1.shape[1]))),axis=0)
    elif rows1 > rows2:
        im2 = concatenate((im2,zeros((rows1-rows2,im2.shape[1]))),axis=0)
    # if none of these cases they are equal, no filling needed.
    # 如果这些情况都没有，那么它们的行数相同，不需要填充
    
    return concatenate((im1,im2), axis=1)
    
    
def plot_matches(im1,im2,locs1,locs2,matchscores,show_below=True):
    """ Show a figure with lines joining the accepted matches 
        input: im1,im2 (images as arrays), locs1,locs2 (feature locations), 
        matchscores (as output from 'match()'), 
        show_below (if images should be shown below matches). """
    """ 显示带有连接匹配之间连线的图片
         输入：im1，im2（数组图像），locs1，locs2（特征位置），
             matchscores（“match()”的输出），
             show_below（如果图像应显示在匹配的下方）。 """
    
    im3 = appendimages(im1,im2)
    if show_below:
        im3 = vstack((im3,im3))
    
    imshow(im3)
    
    cols1 = im1.shape[1]
    for i,m in enumerate(matchscores):
        if m>0:
            plot([locs1[i][1],locs2[m][1]+cols1],[locs1[i][0],locs2[m][0]],'c')
    axis('off')

'''
# 使用归一化互相关矩阵
wid = 5
harrisim = harris.compute_harris_reponse(im1,5)
filtered_coords1 = harris.get_harris_points(harrisim,wid+1)
d1 = harris.get_descriptors(im1,filtered_coords1,wid)

harrisim = harris.compute_harris_reponse(im2,5)
filtered_coords2 = harris.getharris_points(harrisim,wid+1)
d2 = harris.get_descriptors(im2,filtered_coords2,wid)

print('starting matching')
matches = harris.match_twosided(d1,d2)

figure()
gray()
harris.plot_matches(im1,im2,filtered_coords1,filtered_coords2,matches)     # 画出匹配子集 matches[:100]
show()
'''
