close all
clear all
clc
tic

u_exact = double(imread('Cameraman.tif'));%Image

%parameters
nx  = 16; 
beta =0.01;  alpha = 1e-8;
h=0.5;
tol = 1e-8; 
maxit = 200; 
%-------------------Kernel-----------------------------------------------
syms x y f
f=@(x,y) exp(-(x^2+y^2));
O1=ones(nx,1);
X1=zeros(nx,nx);
K1=zeros(nx^2,nx^2);
M = cell(1, nx);
for i = 1:nx
        M{i} = zeros(nx);
end
for k=1:nx
for i=1:nx
    D(i,1)=f((1-k)*h,(i-1)*h);
    T1 = spdiags(D(i,1)*O1, i-1, nx,nx);
    if i==1
        X1=T1;
    end
    M{k}=M{k}+T1+T1';
end
M{k}=M{k}-X1;

DD = repmat({M{k}},1,nx-(k-1));
DDiag = blkdiag(DD{:});

K1(1:end-(k-1)*nx,(k-1)*nx+1:end) = K1(1:end-(k-1)*nx,(k-1)*nx+1:end) + DDiag;
K1((k-1)*nx+1:end,1:end-(k-1)*nx) = K1((k-1)*nx+1:end,1:end-(k-1)*nx)  + DDiag;

end
KK1=K1; 
%-------------------Blurry Image ----------------------------------------
u=imresize(u_exact,[nx nx]); 
z=conv2(KK1,u(:),'valid');
%-------------- Assemble the RHS of the system --------------------------
b1 =conv2(KK1',z','valid'); %k*kz
[b1r b1c]=size(b1'); 
n1 = nx^2;  m1 = 2*nx*(nx-1); 
%------------------------------MC Matrices-----------------------------------------
U1 = zeros(nx,nx); 
[D,C,AD] = computeMat(U1,nx,m1,beta);;
[B] = Bcomp11(nx);
%----------------------Assemble the LHS of the system----------------------
K1=[KK1'*KK1 -alpha*AD;
     zeros(nx^2)  eye(nx^2)];
[br bc]=size(B); 
OC =zeros([br, nx^2]);
B1=[-B OC;
     OC -B;
     OC OC];
B2=[OC' alpha*B' -alpha*B';
     -B'  OC'  OC'];
OD =zeros(br, br);
D1=[D OD  OD;
     OD D  OD;
     -C OD D];
A = [K1, B2; 
      B1, D1];
[Ar Ac]=size(A);
%---------------------Given Matrix-----------------------------------------
b=zeros(Ar, 1);
b(1:b1r,1)=b1;
%---------------------Preconditioner---------------------------------------
restart = 10; 
x0 = zeros(Ar, 1);
gamma=0
P = [K1+gamma*B2*inv(D1)*B1 B2;
    2*B1, D1];
tic
[x, flag, relres, iter, resvec0] = gmres(A, b, restart, tol, maxit, [], [], x0);
fgmres_time = toc;

% Display results
disp('FGMRES Results:')
disp(['Number of FGMRES Iterations: ' num2str(iter)])
disp(['CPU Time for FGMRES: ' num2str(fgmres_time) ' seconds'])
inner_tol = 1e-6; 
inner_maxit = 1000; 
pcg_x0 = zeros(size(P, 1), 1);
tic;
[pcg_x, pcg_flag, pcg_relres, pcg_inner_iter] = pcg(P, b, inner_tol, inner_maxit, [], [], pcg_x0);
pcg_time = toc;
disp('PCG Results (Inner Iterations):')
disp(['Number of PCG Inner Iterations: ' num2str(pcg_inner_iter)])
disp(['CPU Time for PCG: ' num2str(pcg_time) ' seconds'])
rel_error = norm(A*x - b) / norm(b);
disp(['Relative Error: ' num2str(rel_error)])

gamma=0.7
P1 = [K1+gamma*B2*inv(D1)*B1 B2;
    2*B1, D1];
tic
[x, flag, relres, iter, resvec1] = gmres(A, b, restart, tol, maxit, P1, [], x0);
fgmres_time = toc;

% Display results
disp('FGMRES Results:')
disp(['Number of FGMRES Iterations: ' num2str(iter)])
disp(['CPU Time for FGMRES: ' num2str(fgmres_time) ' seconds'])
inner_tol = 1e-6; 
inner_maxit = 1000; 
pcg_x0 = zeros(size(P1, 1), 1);
tic;
[pcg_x, pcg_flag, pcg_relres, pcg_inner_iter] = pcg(P1, b, inner_tol, inner_maxit, [], [], pcg_x0);
pcg_time = toc;
disp('PCG Results (Inner Iterations):')
disp(['Number of PCG Inner Iterations: ' num2str(pcg_inner_iter)])
disp(['CPU Time for PCG: ' num2str(pcg_time) ' seconds'])
rel_error = norm(A*x - b) / norm(b);
disp(['Relative Error: ' num2str(rel_error)])


gamma=0.9
P2 = [K1+gamma*B2*inv(D1)*B1 B2;
    2*B1, D1];
tic
[x, flag, relres, iter, resvec2] = gmres(A, b, restart, tol, maxit, P2, [], x0);
fgmres_time = toc;

% Display results
disp('FGMRES Results:')
disp(['Number of FGMRES Iterations: ' num2str(iter)])
disp(['CPU Time for FGMRES: ' num2str(fgmres_time) ' seconds'])
inner_tol = 1e-6; 
inner_maxit = 1000; 
pcg_x0 = zeros(size(P2, 1), 1);
tic;
[pcg_x, pcg_flag, pcg_relres, pcg_inner_iter] = pcg(P2, b, inner_tol, inner_maxit, [], [], pcg_x0);
pcg_time = toc;
disp('PCG Results (Inner Iterations):')
disp(['Number of PCG Inner Iterations: ' num2str(pcg_inner_iter)])
disp(['CPU Time for PCG: ' num2str(pcg_time) ' seconds'])
rel_error = norm(A*x - b) / norm(b);
disp(['Relative Error: ' num2str(rel_error)])

gamma=1
P3 = [K1+gamma*B2*inv(D1)*B1 B2;
    2*B1, D1];
tic
[x, flag, relres, iter, resvec3] = gmres(A, b, restart, tol, maxit, P3, [], x0);
fgmres_time = toc;

% Display results
disp('FGMRES Results:')
disp(['Number of FGMRES Iterations: ' num2str(iter)])
disp(['CPU Time for FGMRES: ' num2str(fgmres_time) ' seconds'])
inner_tol = 1e-6; 
inner_maxit = 1000; 
pcg_x0 = zeros(size(P3, 1), 1);
tic;
[pcg_x, pcg_flag, pcg_relres, pcg_inner_iter] = pcg(P3, b, inner_tol, inner_maxit, [], [], pcg_x0);
pcg_time = toc;
disp('PCG Results (Inner Iterations):')
disp(['Number of PCG Inner Iterations: ' num2str(pcg_inner_iter)])
disp(['CPU Time for PCG: ' num2str(pcg_time) ' seconds'])
rel_error = norm(A*x - b) / norm(b);
disp(['Relative Error: ' num2str(rel_error)])

gamma=1.2
P4 = [K1+gamma*B2*inv(D1)*B1 B2;
    2*B1, D1];
tic
[x, flag, relres, iter, resvec4] = gmres(A, b, restart, tol, maxit, P4, [], x0);
fgmres_time = toc;

% Display results
disp('FGMRES Results:')
disp(['Number of FGMRES Iterations: ' num2str(iter)])
disp(['CPU Time for FGMRES: ' num2str(fgmres_time) ' seconds'])
inner_tol = 1e-6; 
inner_maxit = 1000; 
pcg_x0 = zeros(size(P4, 1), 1);
tic;
[pcg_x, pcg_flag, pcg_relres, pcg_inner_iter] = pcg(P4, b, inner_tol, inner_maxit, [], [], pcg_x0);
pcg_time = toc;
disp('PCG Results (Inner Iterations):')
disp(['Number of PCG Inner Iterations: ' num2str(pcg_inner_iter)])
disp(['CPU Time for PCG: ' num2str(pcg_time) ' seconds'])
rel_error = norm(A*x - b) / norm(b);
disp(['Relative Error: ' num2str(rel_error)])

gamma=1.4
P5 = [K1+gamma*B2*inv(D1)*B1 B2;
    2*B1, D1];
tic
[x, flag, relres, iter, resvec5] = gmres(A, b, restart, tol, maxit, P5, [], x0);
fgmres_time = toc;

% Display results
disp('FGMRES Results:')
disp(['Number of FGMRES Iterations: ' num2str(iter)])
disp(['CPU Time for FGMRES: ' num2str(fgmres_time) ' seconds'])
inner_tol = 1e-6; 
inner_maxit = 1000; 
pcg_x0 = zeros(size(P5, 1), 1);
tic;
[pcg_x, pcg_flag, pcg_relres, pcg_inner_iter] = pcg(P5, b, inner_tol, inner_maxit, [], [], pcg_x0);
pcg_time = toc;
disp('PCG Results (Inner Iterations):')
disp(['Number of PCG Inner Iterations: ' num2str(pcg_inner_iter)])
disp(['CPU Time for PCG: ' num2str(pcg_time) ' seconds'])
rel_error = norm(A*x - b) / norm(b);
disp(['Relative Error: ' num2str(rel_error)])

gamma=1.6
P6 = [K1+gamma*B2*inv(D1)*B1 B2;
    2*B1, D1];
tic
[x, flag, relres, iter, resvec6] = gmres(A, b, restart, tol, maxit, P6, [], x0);
fgmres_time = toc;

% Display results
disp('FGMRES Results:')
disp(['Number of FGMRES Iterations: ' num2str(iter)])
disp(['CPU Time for FGMRES: ' num2str(fgmres_time) ' seconds'])
inner_tol = 1e-6; 
inner_maxit = 1000; 
pcg_x0 = zeros(size(P6, 1), 1);
tic;
[pcg_x, pcg_flag, pcg_relres, pcg_inner_iter] = pcg(P6, b, inner_tol, inner_maxit, [], [], pcg_x0);
pcg_time = toc;
disp('PCG Results (Inner Iterations):')
disp(['Number of PCG Inner Iterations: ' num2str(pcg_inner_iter)])
disp(['CPU Time for PCG: ' num2str(pcg_time) ' seconds'])
rel_error = norm(A*x - b) / norm(b);
disp(['Relative Error: ' num2str(rel_error)])

gamma=1.7
P7 = [K1+gamma*B2*inv(D1)*B1 B2;
    2*B1, D1];
tic
[x, flag, relres, iter, resvec7] = gmres(A, b, restart, tol, maxit, P7, [], x0);
fgmres_time = toc;

% Display results
disp('FGMRES Results:')
disp(['Number of FGMRES Iterations: ' num2str(iter)])
disp(['CPU Time for FGMRES: ' num2str(fgmres_time) ' seconds'])
inner_tol = 1e-6; 
inner_maxit = 1000; 
pcg_x0 = zeros(size(P7, 1), 1);
tic;
[pcg_x, pcg_flag, pcg_relres, pcg_inner_iter] = pcg(P7, b, inner_tol, inner_maxit, [], [], pcg_x0);
pcg_time = toc;
disp('PCG Results (Inner Iterations):')
disp(['Number of PCG Inner Iterations: ' num2str(pcg_inner_iter)])
disp(['CPU Time for PCG: ' num2str(pcg_time) ' seconds'])
rel_error = norm(A*x - b) / norm(b);
disp(['Relative Error: ' num2str(rel_error)])
%----------------------------Functions-------------------------------------
 function [D,C,A] = computeMat(U,nx,m,beta)
h0=1/nx;
[X,Y] = meshgrid(h0/2:h0:1-h0/2);
% give me U from previouw computations

nn = size(U,1);
UU = sparse(nn+2,nn+2);

% we are using reflection bounday conditions 
% another word, we are using normal boundary condition to be zero
UU(2:nn+1,2:nn+1) = U;
UU(1,:) = UU(2,:);
UU(nn+2,:) = UU(nn+1,:);
UU(:,1) = UU(:,2);
UU(:,nn+2) = UU(:,nn+1);
%------------------ Matrix D ------------------
Uxr = diff(U,1,2)/h0; % x-deriv at red points
xb = h0/2:h0:1-h0/2;   yr=xb;
yb = h0:h0:1-h0;       xr=yb;
[Xb,Yb]=meshgrid(xb,yb);
[Xr,Yr]=meshgrid(xr,yr);
Uxb = interp2(Xr,Yr,Uxr,Xb,Yb,'spline');
 
 Uyb = diff(U,1,1)/h0; % y-deriv at blue points
 Uyr = interp2(Xb,Yb,Uyb,Xr,Yr,'spline');
  
 Dr = sqrt( Uxr.^2 + Uyr.^2 + beta^2 );
 Db = sqrt( Uxb.^2 + Uyb.^2 + beta^2 );
 mm1 = size(Dr,1);
 
 Dvr = Dr(:);  Dvb = Db(:); Dv=[Dvr;Dvb];
 
 siz_u1 =size(Dvr);
 siz_u2 =size(Dvb);
 ddd = [ sparse(m,1) , Dv , sparse(m,1) ];
 D = spdiags(ddd,[-1 0 1],m,m);
 %-------------------------Matrix C----------------------------
 Wxr = diff(UU,3,2)/h0; % x-deriv at red points
 Wyb = diff(UU,3,1)/h0; % y-deriv at blue points
 
Wr = Wxr(1:mm1,:); 
Wb = Wyb(:,1:mm1); 
 
 
 Dwr = (Wr(:).*Uxr(:))./Dr(:);  Dwb = (Wb(:).*Uyb(:))./Db(:); Dw=[Dwr;Dwb];
 
 www = [ sparse(m,1) , Dw , sparse(m,1) ];
 C = spdiags(www,[-1 0 1],m,m);
 
 %-------------------- Matrix A -----------------------------
 
 E = zeros(nx,nx); 
 E(1,1)=1; E(nx,nx)=1;
 M=speye(nx,nx);

 A1 = kron(M,E);
 A2 = kron(E,M);
 A = 2*(A1 + A2)/(beta*h0);

 end
function [B] = Bcomp11(nx)
e = ones(nx,1);
E = spdiags([0*e -1*e e], -1:1, nx, nx);
E1 =E(1:nx-1,:);
 
M1=eye(nx,nx);
B1=kron(E1,M1);
 
E2 = eye(nx);
M2 = spdiags([0*e -1*e e], -1:1, nx-1, nx);
B2 = kron(E2,M2);
 
B = [B1;B2];
end 
 
 













