import random
import numpy as np

import math
import pandas as pd
import time

class network:
    def __init__(self,dice_game,model_folder):
        self.size=20
        self.rate=0.99
        self.dice_game=dice_game
        self.model_folder=model_folder
        self.depth=10
        self.weights=[]
        self.bias=[]
        for i in range(self.depth):
            holder=[]
            
            if i==0:
                for i in range(self.size):
                    array=(np.random.uniform(0,1,len(dice_game)))
                    holder.append(array)
            else:
                for i in range(self.size):
                    array=(np.random.uniform(0,1,self.size))
                    holder.append(array)
            self.bias.append(np.random.uniform(0,1,self.size))
            lemons=np.array(holder).astype(float)
            
            array=np.array(lemons)
            
            self.weights.append(array)
        
        self.lastb=np.random.uniform(0,1,7)
        print(type(self.lastb))
        self.lastw=[]
        for i in range(7):
            self.lastw.append(np.random.uniform(0,1,self.size))
        self.lastw=np.array(self.lastw)
        self.lastw=np.array(self.lastw.astype(float))
        
        
    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))
    def sigdiv(self,x):
        
        return np.exp(-x) / (1 + np.exp(-x))**2
    def main(self,dice_game):
        self.dice_game=np.array(dice_game)
        sig=0
        self.sig=[]
        #self.sig.append(dice_game)
        for i in range(self.depth):
            if i !=0:
                z=[]
                for j in range(self.size):
                    #print(type(sig),type(self.weights[i][j]))
                    hold=np.dot( sig,self.weights[i][j])+self.bias[i][j]
                    z.append(hold)
                lemons=np.array(z).astype(float)
            
                z=np.array(lemons)   
                
                
                sig=self.sigmoid(z)
                
                self.sig.append(sig)
           
            else:
                z=[]
                for j in range(self.size):
                    
                    hold=np.dot(self.dice_game, self.weights[i][j])+self.bias[i][j]
                    
                    z.append(hold)
                lemons=np.array(z).astype(float)
            
                z=np.array(lemons)
                
                sig=self.sigmoid(z)
                
                self.sig.append(sig)
        z=[]
        
        for j in range(7):
            
            z.append(np.dot( self.lastw[j],sig)+self.lastb[j])
        lemons=np.array(z).astype(float)
            
        z=np.array(lemons)
        self.result=self.sigmoid(z)
        
        return self.result
   
    def ajusts(self,expected,dice_game):
        sig=0
        self.dice_game=np.array(dice_game)
        expected=np.array(expected)
        dcda=2*(self.result-expected)
        dw=[]
        db=[]
        ddw=[]
        dab=np.zeros(self.size)
        ddb=[]
        for i in range(len(self.result)):
            z=np.dot( self.lastw[i],self.sig[self.depth-1])+self.lastb[i]
            sig=self.sigdiv(z)
            
            dab+=(np.dot(np.dot(sig,dcda[i]),self.weights[self.depth-1][i]))
            ddb.append(np.dot(sig,dcda[i]))
            
            ddw.append(np.dot(np.dot(self.sig[4],sig),dcda[i]))
        #print(ddw)
        db.append(ddb)
        dw.append(ddw)
        #print(len(self.weights[0][0]))
        for i in range(self.depth):
            i=self.depth-1-i
            #print(i)
            if i ==0:
                dcda=dab
                dab=np.zeros(self.size)
                
                
                ddw=[]
                ddb=[]
                for j in range(self.size):
                    
                    z=np.dot( self.dice_game,self.weights[i][j])+self.bias[i][j]
                    
                    sig=self.sigdiv(z)
                    #dab+=(np.dot(np.dot(sig,self.weights[i][i]),dcda[j]))
                    
                    ddb.append(np.dot(sig,dcda[j]))
                    ddw.append(np.dot(np.dot(sig,self.sig[i-1]),dcda[j]))
                #print(ddw.shape)
                db.append(ddb)
                dw.append(ddw)
            else:
                dcda=dab
                dab=np.zeros(self.size)
                ddb=[]
                
                ddw=[]
                for j in range(self.size):
                    
                    z=np.dot( self.sig[i-1],self.weights[i][j])+self.bias[i][j]
                    sig=self.sigdiv(z)
                    dab+=np.dot(np.dot(sig,dcda[j]),self.weights[i][j])
                    ddb.append(np.dot(sig,dcda[j]))
                    ddw.append(np.dot(np.dot(sig,self.sig[i-1]),dcda[j]))
                #print(ddw.shape)
                db.append(ddb)
                dw.append(ddw)
        
        for i in range(self.depth):
            for j in range(len(dw[i])):
                if i==0:
                     #print(len(dw[i][j]),i)
                     self.lastw[j]=self.lastw[j]-self.rate* dw[i][j]
                     
                else:
                    
                    self.weights[i][j]=self.weights[i][j]-self.rate*dw[i][j]
                    
            if i==0:
                dl=np.array(db[i])
                dl=np.array(dl.astype(float))
                
                self.lastb=self.lastb-(self.rate*dl)
            else:
                dl=np.array(db[i])
                dl=np.array(dl.astype(float))
                self.bias[i]=self.bias[i]-(self.rate*dl)
                
    def finished(self):
        for i in range(self.depth+1):
            if i==self.depth:
                df=pd.DataFrame(self.lastw)
                df.to_csv(f'{self.model_folder}/weight{i}.csv',index=False)
                df=pd.DataFrame(self.lastb)
                df.to_csv(f'{self.model_folder}/bias{i}.csv',index=False)
            else:
                df=pd.DataFrame(self.weights[i])
                df.to_csv(f'{self.model_folder}/weight{i}.csv',index=False)
                df=pd.DataFrame(self.bias[i])
                df.to_csv(f'{self.model_folder}/bias{i}.csv',index=False)
            
    def loading(self):
         for i in range(self.depth+1):
            if i==self.depth:
                df=pd.read_csv(f'{self.model_folder}/weight{i}.csv')
                df=df.to_numpy()
                df=np.array(df)
                self.lastw=df
                df=pd.read_csv(f'{self.model_folder}/bias{i}.csv')
                df=df.to_numpy()
                df=np.array(df)
                self.lastb=df.flatten()
            else:
                df=pd.read_csv(f'{self.model_folder}/weight{i}.csv')
                df=df.to_numpy()
                df=np.array(df)
                self.weights[i]=df
                df=pd.read_csv(f'{self.model_folder}/bias{i}.csv')
                df=df.to_numpy()
                df=np.array(df)
                self.bias[i]=df.flatten()