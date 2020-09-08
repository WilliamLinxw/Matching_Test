import numpy as np
import cv2

class BOW(object):
    
    def __init__(self,):
        #创建一个SIFT对象  用于关键点提取
        self.feature_detector  = cv2.xfeatures2d.SIFT_create()
        #创建一个SIFT对象  用于关键点描述符提取 
        self.descriptor_extractor = cv2.xfeatures2d.SIFT_create()

    def path(self,cls,i):
        '''
        用于获取图片的全路径 
        '''
        return '%s/%s%d.jpg'%(train_path, cls, i + 1)

    def fit(self,train_path,k):
        '''
        开始训练
        
        args：
            train_path：训练集图片路径  我们使用的数据格式为 train_path/dog/dog.i.jpg  train_path/cat/cat.i.jpg
            k：k-means参数k
        '''
        self.train_path = train_path        
        
        #FLANN匹配  参数algorithm用来指定匹配所使用的算法，可以选择的有LinearIndex、KTreeIndex、KMeansIndex、CompositeIndex和AutotuneIndex，这里选择的是KTreeIndex(使用kd树实现最近邻搜索)
        flann_params = dict(algorithm=1,tree=5)
        flann = cv2.FlannBasedMatcher(flann_params,{})
        
        #创建BOW训练器，指定k-means参数k   把处理好的特征数据全部合并，利用聚类把特征词分为若干类，此若干类的数目由自己设定，每一类相当于一个视觉词汇
        bow_kmeans_trainer = cv2.BOWKMeansTrainer(k)
        
        pos = 'pos-'
        neg = 'neg-'
        
        #指定用于提取词汇字典的样本数
        length = 10
        #合并特征数据  每个类从数据集中读取length张图片(length个狗,length个猫)，通过聚类创建视觉词汇
        for i in range(length):
            bow_kmeans_trainer.add(self.sift_descriptor_extractor(self.path(pos,i)))
            bow_kmeans_trainer.add(self.sift_descriptor_extractor(self.path(neg,i)))

        #进行k-means聚类，返回词汇字典 也就是聚类中心
        voc = bow_kmeans_trainer.cluster()
        
        #输出词汇字典  <class 'numpy.ndarray'> (40, 128)
        print(type(voc),voc.shape)
        
        #初始化bow提取器(设置词汇字典),用于提取每一张图像的BOW特征描述
        self.bow_img_descriptor_extractor = cv2.BOWImgDescriptorExtractor(self.descriptor_extractor,flann)        
        self.bow_img_descriptor_extractor.setVocabulary(voc)
        
        
        #创建两个数组，分别对应训练数据和标签，并用BOWImgDescriptorExtractor产生的描述符填充
        #按照下面的方法生成相应的正负样本图片的标签 1：正匹配  -1：负匹配
        traindata,trainlabels = [],[]
        for i in range(20):   #这里取200张图像做训练
            traindata.extend(self.bow_descriptor_extractor(self.path(pos,i)))
            trainlabels.append(1)
            traindata.extend(self.bow_descriptor_extractor(self.path(neg,i)))
            trainlabels.append(-1)
         
        #创建一个SVM对象    
        self.svm = cv2.ml.SVM_create()
        #使用训练数据和标签进行训练
        self.svm.train(np.array(traindata),cv2.ml.ROW_SAMPLE,np.array(trainlabels))


    def predict(self,img_path):
        '''
        进行预测样本   
        '''
        #提取图片的BOW特征描述
        data = self.bow_descriptor_extractor(img_path)
        res = self.svm.predict(data)
        print(img_path,'\t',res[1][0][0])
        
        #如果是狗 返回True
        if res[1][0][0] == 1.0:
            return True
        #如果是猫，返回False
        else:
            return False
        
    
     
    def sift_descriptor_extractor(self,img_path):
        '''
        特征提取：提取数据集中每幅图像的特征点，然后提取特征描述符，形成特征数据(如：SIFT或者SURF方法)；
        '''
        im = cv2.imread(img_path,0)     
        return self.descriptor_extractor.compute(im,self.feature_detector.detect(im))[1]  
    
    
    def bow_descriptor_extractor(self,img_path):
        '''
        提取图像的BOW特征描述(即利用视觉词袋量化图像特征)
        '''
        im = cv2.imread(img_path,0)
        return self.bow_img_descriptor_extractor.compute(im,self.feature_detector.detect(im))
    


if __name__ == '__main__':
    # #测试样本数量，测试结果
    # test_samples = 100
    # test_results = np.zeros(test_samples,dtype=np.bool)
    
    #训练集图片路径  狗和猫两类  进行训练
    train_path = '/media/caesarlin/DATA/Cars_Images/cars_train'
    bow = BOW()
    bow.fit(train_path,40)
    
    
    # #指定测试图像路径
    # for index in range(test_samples):
    #     dog = './data/cat_and_dog/data/train/dog/dog.{0}.jpg'.format(index)
    #     dog_img = cv2.imread(dog)    
    
    #     #预测
    #     dog_predict = bow.predict(dog)    
    #     test_results[index] = dog_predict

    car, notcar = '/media/caesarlin/DATA/Cars_Images/Car.jpg', '/media/caesarlin/DATA/Cars_Images/Soccer.jpg'
    car_img = cv2.imread(car)
    notcar_img = cv2.imread(notcar)
        
    car_predict = bow.predict(car)
    not_car_predict = bow.predict(notcar)
        
    font = cv2.FONT_HERSHEY_SIMPLEX

    if car_predict == True:
        cv2.putText(car_img, 'Car Detected', (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    if not_car_predict == False:
        cv2.putText(notcar_img, 'Car Not Detected', (10, 30), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('BOW + SVM Success', car_img)
    cv2.imshow('BOW + SVM Failure', notcar_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()