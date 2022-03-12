import cv2
import random
import numpy as np
import copy
import skimage.measure
import os
from rede_1 import*
class RedeConv:
	def __init__(self,):
		self.kernels=[]
		self.img_atual=[]
		self.imgs=[]
		self.ativador=RedeConv.relu
	
	def tanh(self,x):
		return math.tanh(x)
	@staticmethod
	def relu(x):
		if(x>0):
			return x
		else:
			return 0
	def limpa(self):
		self.imgs=[]
	def salvar(self,nome,mat):
	    with open(nome+'.txt', 'w') as arquivo:
	        for i in range(len(mat)):
	            for j in range(len(mat[0])):
	                arquivo.write(str(mat[i][j])+' ')
	            arquivo.write('\n')
	def salvarKernel(self):
		for ic,i in enumerate(self.kernels):
			os.system("mkdir filtro_"+str(ic))
			for cc,c in enumerate(i):
				self.salvar("mkdir filtro_"+str(cc),c)
			

	def addKernel(self,tam=0,qtd=0,kn=[]):
		aux_kn=[]
		if(kn!=[]):
			aux_kn=kn
		else:
			for c in range(qtd):
				kn=[]
				for i in range(tam):
					aux=[]
					for j in range(tam):

						aux.append((random.randint(-5,5)))
					kn.append(aux)
				aux_kn.append(kn)
		self.kernels.append(aux_kn)
	def runKernel(self,index):
		img_branco=[]
		img_kn=[]
		lenKn=len(self.kernels[index][0])-1
		if(self.imgs==[]):
			img_kn=[self.img_atual]
		else:
			#print("d")
			img_kn=self.imgs

		for i in img_kn:
			height = i.shape[0]
			width = i.shape[1]
			for j in self.kernels[index]:
				img_branco.append(np.zeros((height-lenKn,width-lenKn,1), np.uint8))
			#print(len(img_branco))
		for cim,im in enumerate(img_kn):
			#print(im.shape)
			height = im.shape[0]
			width = im.shape[1]
			
			for ckn,kn in enumerate(self.kernels[index]):
				lenKn=len(kn)-1
			
				for i in range(height-lenKn):
					for j in range(height-lenKn):
						aux=[]
						for c in range(lenKn+1):
							for z in range(lenKn+1):
								aux.append(kn[c][z]*im[c+i][z+j])
						img_branco[ckn+cim][i][j]=self.ativador(sum(aux) / float(len(aux)) )

		self.imgs=copy.deepcopy(img_branco)

		
		#print(len(self.imgs))
		#print(len(self.kernels[index]))
		#print(len(img_branco))
		
	def addBorda(self,img):
		return cv2.copyMakeBorder(img,1,0,1,0,cv2.BORDER_CONSTANT)
	def maxPool(self, kernel= 2):
		#print("ini")
		img_kn=[]
		if(self.imgs==[]):
			img_kn=[self.img_atual]
		else:
			#print("d")
			img_kn=self.imgs
		aux_img_pronto=[]
		for im in img_kn:
			inix=0
			iniy=0
			im_b=copy.deepcopy(im)
			while True:
				#print(len(im)%kernel)
				#print(iniy)
				if((len(im_b)%kernel)!=0):
					#print(len(im)%kernel)
					im_b=self.addBorda(im)
				else:
					break
				
			#print("f")
			img2=np.zeros((int(len(im_b)/kernel),int(len(im_b)/kernel)))
			#print(img2.shape)
			listapixel=[]
			while True:
				#print(iniy,inix,len(im_b))

				if(iniy>=len(im_b)):

					inix+=kernel
					iniy=0
					#print(len(listapixel))
					if(inix>=len(im_b)):
						break
					for i in range(len(img2)):

						img2[int(inix/kernel)][i]=listapixel[i]

					listapixel=[]
					

				aux=0
				
				for i in range(kernel):
					for j in range(kernel):
						if(im_b[i+inix][j+iniy]>aux):
							aux=im_b[i+inix][j+iniy]

				listapixel.append(aux)

				iniy+=kernel

			
					
			aux_img_pronto.append(img2)
			#print("fim")
		self.imgs=copy.deepcopy(aux_img_pronto)

	def alinhaPixel(self):
		img_kn=[]
		if(self.imgs==[]):
			img_kn=[self.img_atual]
		else:
			#print("d")
			img_kn=self.imgs

		aux=[]
		for i in img_kn:
			for c in i:
				for x in c:
					aux.append(self.tanh(x))
		return aux
	def showImg(self):
		for c,i in enumerate(self.imgs):
			cv2.imshow('image{}'.format(c),i)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
	def carregaImg(self,img):
		self.img_atual=resized = cv2.resize(cv2.imread(img, cv2.IMREAD_GRAYSCALE), (200,200), interpolation = cv2.INTER_AREA)
		



			