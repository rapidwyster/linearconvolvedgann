import random
import math
def FIT(X,Y,N):
	fout = PWS(N(X),Y)
	return sum(DOT(fout,fout))
def LCheck(LC):
	try:
		return len(LC)
	except:
		return 0
def DCheck(X,Y):
	if Y == 0:
		return 0-X
	else:
		return X/Y
def ACTIVATE(ATI):
	return 1/(1+math.e**-ATI)
def DOT(X,Y):
	return [X[F]*Y[F] for F in range(min(LCheck(X),LCheck(Y)))]
def PWS(X,Y):
	return [Y[F]-X[F] for F in range(min(LCheck(X),LCheck(Y)))]
def PWA(X,Y):
	return [Y[F]+X[F] for F in range(min(LCheck(X),LCheck(Y)))]
def SCM(X,Y):
	return [X*y for y in Y]
def SCA(X,Y):
	return [X+y for y in Y]
def SCD(X,Y):
	return [DCheck(y,X) for y in Y]
class Memory:
	def __init__(self):
		self.set_pre(None)
		self.set_post(None)
	def set_pre(self,PRE):
		self.pre = PRE
	def set_post(self,POST):
		self.post = POST
	def get_pre(self):
		return self.pre
	def get_post(self):
		return self.post
class Neuron(Memory):
	def __init__(self,N=1,LR=1):
		self.set_bias(random.random())
		self.set_weights([random.random() for n in range(N)])
		self.set_lr(LR)
	def set_lr(self,LR):
		self.lr = LR
	def get_lr(self):
		return self.lr
	def set_bias(self,BIAS):
		self.bias = BIAS
	def set_weights(self,WEIGHTS):
		self.weights = WEIGHTS
	def get_bias(self):
		return self.bias
	def get_weights(self):
		return self.weights
	def __repr__(self):
		return '{}'.format(self)
	def __str__(self):
		return '{}+{}'.format(self.get_weights(),self.get_bias())
	def __call__(self,other):
		self.set_pre(other)
		self.set_post(ACTIVATE(sum(DOT(self.get_weights(),self.get_pre()))+self.get_bias()))
		return self.get_post()
	def get_dxb(self,other,desired):
		dxz = self(other)*(1-self.get_post())
		return dxz*2*(self.get_post()-desired)
	def get_dxw(self,other,desired):
		return SCM(self.get_dxb(other,desired),self.get_pre())
	def get_learn(self,other,desired):
		self.set_bias(self.get_bias()-self.get_dxb(other,desired)*self.get_lr())
		self.set_weights(PWS(SCM(self.get_lr(),self.get_dxw(other,desired)),self.get_weights()))
		return self
class Layer(Neuron):
	def __init__(self,M=2,N=2):
		Memory.__init__(self)
		self.set_layer([Neuron(M) for n in range(N)])
	def set_layer(self,LAYER):
		self.layer = LAYER
	def get_layer(self):
		return self.layer
	def __str__(self):
		return '{}'.format(self.get_layer())
	def __repr__(self):
		return '{}'.format(self)
	def __len__(self):
		return LCheck(self.get_layer())
	def __call__(self,other):
		self.set_pre(other)
		self.set_post([F(self.get_pre()) for F in self.get_layer()])
		return self.get_post()
	def get_learn(self,other,desired):
		error = PWS(self(other),desired)
		layer = self.get_layer()
		self.set_layer([layer[n].get_learn(layer[n].get_pre(),layer[n].get_post()+error[n]) for n in range(len(self))])
		return self
	def get_error(self,other,desired):
		return FIT(other,desired,self)
	
class Network(Layer):
	def __init__(self,M=2,N=2,D=3):
		Memory.__init__(self)
		self.set_network([Layer(M,N),*[Layer(N,N) for n in range(D-1)]])
	def set_network(self,NETWORK):
		self.layer = NETWORK
	def get_network(self):
		return self.layer
	def __repr__(self):
		return '{}'.format(self)
	def __str__(self):
		return '{}'.format(self.get_network())
	def __call__(self,other):
		self.set_pre(other)
		self.set_post(self.get_network()[0](self.get_pre()))
		for L in self.get_network()[1:]:
			self.set_post(L(self.get_post()))
		return self.get_post()
	def get_learn(self,other,desired):
		error = PWS(self(other),desired)
		network = self.get_network()
		for N in reversed(network):
			N.get_learn(N.get_pre(),PWA(N.get_post(),error))
		self.set_network(network)
		return self
class Descriminator(Network):
	def __init__(self,M=2,N=2,D=3):
		super().__init__(M,N,D)
		self.layer.append(Layer(N,1))
	def get_learn(self,other,desired):
		error = desired[0]-self(other)[0]
		network = self.get_network()
		for N in reversed(network):
			N.get_learn(N.get_pre(),SCA(error,N.get_post()))
		self.set_network(network)
		return self
def Padd(X):
	try:
		return [0,*X,0]
	except:
		return [0,X,0]
def Slide(X,Y):
	return [sum(DOT(X,Y[N:])) for N in range(len(Y)-len(X)+1)]
def Convolve(X,Y):
	return Slide(X,Padd(Y))
class CNeuron(Neuron):
	def __init__(self,N=4,LR=1):
		super().__init__(N,LR)
	def __call__(self,other):
		self.set_pre(other)
		self.set_post(ACTIVATE(sum(Convolve(self.get_weights(),self.get_pre()))))
		return self.get_post()
class CLayer(Layer):
	def __init__(self,M,N):
		Memory.__init__(self)
		self.set_layer([CNeuron(M) for n in range(N)])
class CNetwork(Network):
	def __init__(self,M=2,N=3,D=3):
		Memory.__init__(self)
		self.set_network([CLayer(M,N),*[CLayer(N,N) for n in range(D-1)]])
class CDescriminator(Descriminator):
	def __init__(self,M=2,N=2,D=3):
		Memory.__init__(self)
		self.set_network([CLayer(M,N),*[CLayer(N,N) for n in range(D-1)],Layer(N,1)])
		
a = CDescriminator()
b = CNetwork()
f = [0,3]
t = [1]
print(a(f))
for step in range(2000):
	gx = [random.random() for F in f]
	gy = b(gx)
	derror = [0-a(gy)[0]]
	gerror = [1-a(gy)[0] for L in range(len(b))]
	a.get_learn(f,t)
	a.get_learn(gy,derror)
	b.get_learn(gx,gerror)
	a.get_learn([1,1],t)
	gx = [random.random() for F in f]
	gy = b(gx)
	derror = [0-a(gy)[0]]
	gerror = [1-a(gy)[0] for L in range(len(b))]
	a.get_learn(gy,derror)
	b.get_learn(gx,gerror)
	a.get_learn([2,4],t)
	gx = [random.random() for F in f]
	gy = b(gx)
	derror = [0-a(gy)[0]]
	gerror = [1-a(gy)[0] for L in range(len(b))]
	a.get_learn(gy,derror)
	b.get_learn(gx,gerror)
	print(a(b(f)))
print(a(f),a([1,1]),a([2,4]))
