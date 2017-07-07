import numpy as np
import numpy.random as ram

ram.seed(1234)

def alias_setup(probs):
    K= len(probs)
    q=np.zeros(K)
    J=np.zeros(K,dtype=np.int32)

    # Sort the data into the outcomes with probabilities
    # that are larger and smaller than 1/K;
    smaller=[]
    larger=[]
    for idx,probs in enumerate(probs):
        q[idx]=K*probs
        if q[idx]<1.0:
            smaller.append(idx)
        else:
            larger.append(idx)


    # Loop though and create little binary mixtures that
    # appropriately allocate the laeger outcomes over the
    # overall uniform mixture.
    while len(smaller)>0 and len(larger)>0:
        small=smaller.pop()
        large=larger.pop()

        J[small]=large
        q[large]=q[large]-(1.0-q[small])

        if q[large]<1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J,q


def alias_draw(J,q):
    K=len(J)

    # Draw from the overall uniform mixture.
    kk=int(np.floor(ram.rand()*K))

    # Draw from the binary mixture, either keeping the
    # small one, or choosing the associated larger one.
    if ram.rand()<q[kk]:
        return kk
    else:
        return J[kk]

import time
def sampler():
    K=8000000
    N=2000

    # Generating a random probability vector.
    probs=ram.dirichlet(np.ones(K),1).ravel()


    # Construct the table.
    J,q=alias_setup(probs)

    # Generate variates.
    start=time.time()
    X=np.zeros(N)
    for idx in xrange(N):
        X[idx]=alias_draw(J,q)
    print time.time()-start
    start = time.time()
    results=ram.choice(len(probs),N,p=probs)
    print time.time() - start



if __name__ == '__main__':
    sampler()