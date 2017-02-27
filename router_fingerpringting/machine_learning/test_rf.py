from sklearn.datasets import make_blobs
from matplotlib import pyplot
import random
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss




x = [[random.randint(0,3),random.randint(0,3),random.randint(0,3)] for i in range(500)]
y = [[random.randint(7,10),random.randint(7,10),random.randint(7,10)] for i in range(500)]
x_l = [0for i in range(500)]
y_l = [1for i in range(500)]

p = x+y
p_l = x_l + y_l




def random_re(p,p_l)
	pp = []
	pp_l = []
	for i in range(len(p)):
		num = random.randint(0,len(p)-i-1)
		pp.append(p.pop(num))
		pp_l.append(p_l.pop(num))
	return pp,pp_l

X_train, y_train = X[:600], y[:600]
X_valid, y_valid = X[600:800], y[600:800]
X_train_valid, y_train_valid = X[:800], y[:800]
X_test, y_test = X[800:], y[800:]


clf = RandomForestClassifier(n_estimators=25)
clf.fit(X_train_valid, y_train_valid)
clf_probs = clf.predict_proba(X_test)
score = log_loss(y_test, clf_probs)
