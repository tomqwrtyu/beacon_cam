#!/usr/bin/env python3

import os
import cv2 
import glob
import numpy as np
import matplotlib.pyplot as plt

def getData():
    def getMeanAndThresh(color):
        L = np.array([])
        A = np.array([])
        B = np.array([])
        current_dir = os.path.dirname(os.path.abspath(__file__))
        dir = os.path.join(current_dir, 'imgsrc', color)
        path_list = os.listdir(dir)

        for filename in path_list:
            img = cv2.imread("imgsrc/"+color+"/"+filename)
            imgLAB = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l = imgLAB[:,:,0] 
            l = l.reshape(l.shape[0]*l.shape[1]) 
            a = imgLAB[:,:,1] 
            a = a.reshape(a.shape[0]*a.shape[1]) 
            b = imgLAB[:,:,2] 
            b = b.reshape(b.shape[0]*b.shape[1]) 
            L = np.append(L,l) 
            A = np.append(A,a) 
            B = np.append(B,b)
        meanA = round(np.mean(A),0)
        threshA = round(2.58*np.std(A, ddof=1),0)
        meanB = round(np.mean(B),0)
        threshB = round(2.58*np.std(B, ddof=1),0)
        """
        nbins = 10 
        plt.hist2d(A, B, bins=nbins, norm=matplotlib.colors.LogNorm()) 
        plt.title('red')
        plt.xlabel('A') 
        plt.ylabel('B') 
        plt.xlim([0,255]) 
        plt.ylim([0,255])
        plt.show()
        """
        return [color,meanA,threshA,meanB,threshB]

    red = [x for x in getMeanAndThresh('red')]
    green = [x for x in getMeanAndThresh('green')]
    
    with open('cr.txt','w') as cr:
        for item in red:
            cr.write(str(item)+" ")
        cr.write("\n")
        for item in green:
            cr.write(str(item)+" ")
        cr.write("\n")
        
if __name__ == '__main__':
    getData()


