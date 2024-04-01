close all
clear all
clc
tic

nx1  = 16; % Resize to reduce Problem


beta_N =0.01;%beta
alpha_N = 1e-8; %alpha

u_exact1 = double(imread('Cameraman.tif'));
u=imresize(u_exact1,[nx1 nx1]); 

%-------------------Kernel-----------------------------------------------
 N = nx1; h=0.5;
syms x y f
f=@(x,y) exp(-(x^2+y^2));%Gaussion Blur

D1=zeros(N,1); 
O1=ones(N,1);
X1=zeros(N,N);
K1=zeros(N^2,N^2);


M = cell(1, N);
for i = 1:N
        M{i} = zeros(N);
end


for k=1:N
for i=1:N
    D(i,1)=f((1-k)*h,(i-1)*h);
    T1 = spdiags(D(i,1)*O1, i-1, N,N);
    if i==1
        X1=T1;
    end
    M{k}=M{k}+T1+T1';
end
M{k}=M{k}-X1;



DD = repmat({M{k}},1,N-(k-1));
DDiag = blkdiag(DD{:});


K1(1:end-(k-1)*N,(k-1)*N+1:end) = K1(1:end-(k-1)*N,(k-1)*N+1:end) + DDiag;
K1((k-1)*N+1:end,1:end-(k-1)*N) = K1((k-1)*N+1:end,1:end-(k-1)*N)  + DDiag;

end
KK1=K1; 
 
%-------------------Blurry Image ----------------------------------------

z1=conv2(KK1,u(:),'valid');
Blur_psnr = PSNR(z1,u)

%-------------- Assemble the rhd -----------------------------------------

b1 =conv2(KK1',z1','valid'); %k*kz
 
 
n1 = nx1^2;  m1 = 2*nx1*(nx1-1); 
M=speye(n1,n1);

%------------------------------MC-----------------------------------------
U1 = zeros(nx1,nx1); 
[D,C,AD] = computeMat(U1,nx1,m1,beta_N);

[B] = Bcomp11(nx1);
%---------------------Eigenvalues-------------------------------------------

 K1=[KK1'*KK1 -alpha_N*AD;
     zeros(nx1^2)  eye(nx1^2)];
[br bc]=size(B); 
 
OC =zeros([br, nx1^2]);
 B1=[-B OC;
     OC -B;
     OC OC];
 B2=[OC' alpha_N*B' -alpha_N*B';
     -B'  OC'  OC'];
 
 OD =zeros(br, br);

 D1=[D OD  OD;
     OD D  OD;
     -C OD D];
 
  
 A = [K1, B2;
      B1, D1];


eigenvalues_A = eig(full(A));
plot((real(eigenvalues_A)),imag(eigenvalues_A),'o')
title('Eigenvalues of Matrix A ');
xlabel('Real Part');
ylabel('Imaginary Part');
%xlim([-1 20])
grid on;

%-----------------------------Preconditioner---------------------------------------
gamma=1
P = [K1+gamma*B2*inv(D1)*B1 B2;
    2*B1, D1];

AAAAAAAA=inv(P)*A;
%  % Ensure real part is non-negative
%  A = A - min(real(eig(A))) * eye(size(A));
figure;
eigenvalues_A = eig(full(AAAAAAAA));
plot((real(eigenvalues_A)),imag(eigenvalues_A),'o')
title('Eigenvalues of Matrix P^{-1}A ');
xlabel('Real Part');
ylabel('Imaginary Part');
%xlim([0 4])
%title('gamma = 0.9')
grid on;
%--------------------------------------------------------------------
 function [D,C,A] = computeMat(U,nx,m,beta);
h0=1/nx;
[X,Y] = meshgrid(h0/2:h0:1-h0/2);
% U = [20 26 100;5 10 30;25 30 40]  % give me U from previouw computations

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
  
%  siz_Uxr = size(Uxr)
%  siz_Uyb = size(Uyb)
 
 
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
 
% siz_Vr = size(Wr)
% siz_Vb = size(Wb)
 
 
 
 Dwr = (Wr(:).*Uxr(:))./Dr(:);  Dwb = (Wb(:).*Uyb(:))./Db(:); Dw=[Dwr;Dwb];
%  siz_v1 = size(Dwr)
%  siz_v2 = size(Dvb)
 
 www = [ sparse(m,1) , Dw , sparse(m,1) ];
 C = spdiags(www,[-1 0 1],m,m);
 
 %-------------------- Matrix A -----------------------------
 
 E = zeros(nx,nx); 
 E(1,1)=1; E(nx,nx)=1;
 M=speye(nx,nx);

 A1 = kron(M,E);
 A2 = kron(E,M);
 A = 2*(A1 + A2)/(beta*h0);
 %-----------------------------------------------------------
 
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
% L = B'*D*B;
end 
 
 
 
 
 %------------------------------------------------------------
 function p = PSNR(x,y)

% psnr - compute the Peack Signal to Noise Ratio, defined by :
%       PSNR(x,y) = 10*log10( max(max(x),max(y))^2 / |x-y|^2 ).
%
%   p = psnr(x,y);
%
%   Copyright (c) 2004 Gabriel Peyr

d = mean( mean( (x(:)-y(:)).^2 ) );
m1 = max( abs(x(:)) );
m2 = max( abs(y(:)) );
m = max(m1,m2);

p = 10*log10( m^2/d );
 end