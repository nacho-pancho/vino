import numpy as np

def lasso_cost(A,y,theta,x):
    return 0.5*np.sum(np.square(np.dot(A,x)-y))+theta*np.sum(np.abs(x))


def lasso_admm(A,y,theta,l=1):
    m,p = A.shape
    xprev = np.zeros(p)
    x     = np.zeros(p)
    z     = np.zeros(p)
    u     = np.zeros(p)
    Aty = np.dot(A.T,y)
    AtA = np.dot(A.T,A)
    AtApl = AtA + (1/l)*np.eye(p)
    iAtApl = np.linalg.inv(AtApl)
    for K in range(100):
        #
        # 1) x^{k+1} = prox_{lf}(z^k-u^k) = prox_{lf}(w) 
        #            = arg min_x (1/2)||Ax-y||_2^2 + (1/2l)||x-w||_2^2
        #            At*A*x^{k+1} -At*y +(1/l)(x-w) = 0
        #            (AtA + I/l)x^{k+1} = Aty+w/l
        #
        np.copyto(xprev,x)
        x = np.dot(iAtApl,Aty+(z-u)*(1/l))
        if K > 0 and K % 10 == 0:
            dx = np.linalg.norm(x-xprev)/np.linalg.norm(x)
            J = lasso_cost(A,y,theta,x)
            print(K,J,dx)
            if dx < 1e-5:
                break
        # 2) z^{k+1} = prox_{lg}(x^{k+1}+u^k) = prox_lg(w)
        #            = arg min_z theta||z||_1 + (1/2l)||z-w||_2^2
        #            = prox_{(theta*l)||.||_1}(w)
        z = np.minimum(x + u + l*theta, np.maximum(0,x + u - l*theta)) #  FALTA TERMINAR
        #
        # 3) u^{k+1} = u^k + x^{k+1} - z^{k+1}
        #
        u += x - z

    return x


def lasso_admm_batch(A,Y,theta=0.1,l=1, X = None):
    m,p = A.shape
    _,n = Y.shape
    if X is None:
        X     = np.zeros((p,n))
    xprev = np.zeros(p)
    x     = np.zeros(p)
    z     = np.zeros(p)
    u     = np.zeros(p)
    AtY = np.dot(A.T,Y)
    AtA = np.dot(A.T,A)
    AtApl = AtA + (1/l)*np.eye(p)
    iAtApl = np.linalg.inv(AtApl)
    for j in range(n):
        x[:]  = X[:,j] # warm restart, if X was specified
        z[:]  = 0
        u[:]  = 0
        Aty = AtY[:,j]
        y = Y[:,j]
        for K in range(100):
            #
            # 1) x^{k+1} = prox_{lf}(z^k-u^k) = prox_{lf}(w) 
            #            = arg min_x (1/2)||Ax-y||_2^2 + (1/2l)||x-w||_2^2
            #            At*A*x^{k+1} -At*y +(1/l)(x-w) = 0
            #            (AtA + I/l)x^{k+1} = Aty+w/l
            #
            np.copyto(xprev,x)
            x = np.dot(iAtApl,Aty+(z-u)*(1/l))
            if K > 0 and K % 10 == 0:
                dx = np.linalg.norm(x-xprev)/np.linalg.norm(x)
                #J = lasso_cost(A,y,theta,x)
                #print(K,J,dx)
                if dx < 1e-5:
                    break
            # 2) z^{k+1} = prox_{lg}(x^{k+1}+u^k) = prox_lg(w)
            #            = arg min_z theta||z||_1 + (1/2l)||z-w||_2^2
            #            = prox_{(theta*l)||.||_1}(w)
            z = np.minimum(x + u + l*theta, np.maximum(0,x + u - l*theta)) 
            #
            # 3) u^{k+1} = u^k + x^{k+1} - z^{k+1}
            #
            u += x - z
        X[:,j] = x
    return X

def lasso_cd(A,y,theta):
    m,p = A.shape
    x = np.zeros(p)
    G = np.dot(A.T,A)
    g = np.dot(A.T,y)
    
    def cd_iter(G,g,theta,x,i):
        g += x[i]*G[:,i]
        x[i] = (1.0/G[i,i])*min(max(0,g[i]-theta),g[i]+theta)
        g -= x[i]*G[:,i]
        return x,g
    xprev = np.copy(x)
    for K in range(10000):
        np.copyto(xprev,x)
        x,g = cd_iter(G,g,theta,x,K % p)
        if (K % 200) == 0:
            J = lasso_cost(A,y,theta,x)
            dx = np.linalg.norm(x-xprev)/np.linalg.norm(x)            
            print(K,J,dx)
            if dx < 1e-5:
                break
    return x


def normalize_dict(A):
    nA = 1.0/(np.maximum(np.linalg.norm(A,axis=0),1e-4))
    A *= np.outer(np.ones(m),nA)
    return A

def update_dict(A, X, Y, alpha=1e-3):
    m,p = A.shape
    A -= alpha*np.dot((np.dot(A,X)-Y),X.T) # (AX-Y)^2 => (AX-Y)Xt = AXXt - YXt
    normalize_dict(A)


def update_coeffs(A, X, Y):
    m,p = A.shape
    _,n = X.shape        
    return lasso_admm_batch(A, Y, 0.1, l = 1, X=X)


def train_dict(A, Y, lasso_penalty, alpha=1e-3, X=None):
    m,p = A.shape
    _,n = Y.shape
    if X is None:
        X = np.zeros(p,n)
    b = 4*max(m,p)
    for j in range(0,n,b):
        update_coeffs(A,X[:,j:(j+b)], Y[:,j:(j+b)])
        update_dict(A,X[:,j:(j+b)], Y[:,j:(j+b)], alpha=alpha)
        print(j,lasso_cost(A,Y,lasso_penalty,X))




def show_bw(D,margin,bgcolor):
    p, m = D.shape
    minD = np.min(D)
    maxD = np.max(D)
    sD = 1.0/(maxD-minD)
    w = int(np.sqrt( m ))
    mg = int( np.sqrt(p) )
    ng = int(np.ceil( p / mg ))
    Ng = ng*w + (ng+1)*margin
    Mg = mg*w + (mg+1)*margin
    im = bgcolor*np.ones((Mg,Ng))
    k = 0
    for ig  in range(mg):
        for jg in range(ng):
            i0 = margin + ig*(w+margin)
            j0 = margin + jg*(w+margin)
            i1 = i0 + w
            j1 = j0 + w
            atom = np.reshape(D[k,:],(w,w))
            im[i0:i1,j0:j1] = sD*(atom - minD)
            k = k + 1
            if k >= p:
                return im
    return im

def show_color(D,margin,bgcolor):
    p, m = D.shape
    minD = np.min(D)
    maxD = np.max(D)
    sD = 1.0/(maxD-minD)
    w = int(np.sqrt( m/3 ))
    mg = int( np.sqrt(p) )
    ng = int(np.ceil( p / mg ))
    Ng = ng*w + (ng+1)*margin
    Mg = mg*w + (mg+1)*margin
    im = bgcolor*np.ones((Mg,Ng,3))
    k = 0
    for ig  in range(mg):
        for jg in range(ng):
            i0 = margin + ig*(w+margin)
            j0 = margin + jg*(w+margin)
            i1 = i0 + w
            j1 = j0 + w
            atom = np.reshape(D[k,:],(w,w,3))
            im[i0:i1,j0:j1,:] = sD*(atom - minD)
            k = k + 1
            if k >= p:
                return im
    return im
