close all
clear all
clc
tic
u_exact = double(imread('Cameraman.tif'));%Image

%parameters
nx  = 64; 
gama = 1;
beta =1;  alpha = 1e-8;
h=0.5;
tol = 1e-8; 
maxit = 400; 
n = nx^2;  m = 2*nx*(nx-1);  nm = n + m;

N=size(u_exact,1); 
kernel=kernelG(N,300,5); 
kernel=imresize(kernel,[nx nx]);     
ny = nx; hx = 1 / nx; hy = 1 / ny; N=nx; 
kernel=kernel/sum(kernel(:));
m2 = 2*nx; nd2 = nx / 2; kernele = zeros(m2, m2) ;
kernele(nd2+1:nx+nd2,nd2+1:nx+nd2) = kernel ; %extention kernel
k_hat = fft2(fftshift(kernele)) ; clear kernele

u_exact=imresize(u_exact,[nx nx]); 
figure;   imagesc(u_exact);colormap(gray); s=sprintf('Exact image');s=title(s);  
z = integral_cgm(u_exact,k_hat,nx,nx);
Blur_psnr = ppsnr(z,u_exact)
figure;  imagesc(z);colormap(gray);  ss=sprintf('Blured image');ss=title(ss); 
%-------------- assemble the RHD of the system -----------------------
b2 = integral_cgm(z,conj(k_hat),nx,nx); 
b = b2(:);
M=speye(n,n);
%----------------assemble the RHD of the system-----------------------------------------
U = zeros(nx,nx); 
[B] = Bcomp11(nx);

[D,C,A] = computeMat(U,nx,m,beta);
BD = B'*inv(D) ;
L1 = BD*B;
L2 = BD*C*inv(D)*B;
L3 = A*BD*B;
L = (L1)^2 + L2 + L3  ;
L4 = (L1)^2 + L2 ;
[u,flag,RELRES,iter] =  gmres(@(x)KKalphaI(nx,x,k_hat,L,alpha),b,10,tol,maxit);
u1 = reshape(u,[nx nx]);
figure;  imagesc(u1);colormap(gray); ss=sprintf('Deblurred image by GMRES');ss=title(ss); 
%---------------------------- P1 ----------------------------------------- 
c_hat = fft2(kernel, nx, nx); 
[y,flag1,RELRES1,iter1]= gmres(@(x)KKalphaI(nx,x,k_hat,L,alpha),b,10,tol,maxit,@(x)PRE(nx,x,c_hat,A,B,C,D,alpha,gama));
y = reshape(y,nx,nx); 
figure;  imagesc(y);colormap(gray);ss=sprintf('Deblurred image by PGMRES');ss=title(ss);
Deblurred_psnr_by_GMRES = ppsnr(u1,u_exact)
Deblurred_PSNR_by_PGMRES = ppsnr(y,u_exact)
%----------------------------Functions-------------------------------------
function K = kernelG(n, tau, radi);
if nargin<1,help ke_gen;return; end
if nargin<2, tau=200; end
if nargin<3, radi=4; end
K=zeros(n);
R=n/2; h=1/n; h2=h^2;
%RR=n^2/radi+1; 
RR=radi^2;

if radi>0 %___________________________________________

for j=1:n
  for k=1:n
    v=(j-R)^2+(k-R)^2;
    if v <= RR,
      K(j,k)=exp(-v/4/tau^2);
    end;
  end;
end;
sw=sum(K(:));
K=K/sw; %*tau/pi;

else radi<0 %___________________________________________
 range=R-2:R+2;
 K(range,range)=1/25;
end
end   

function Ku = integral_cgm(u,k_hat,nux,nuy) 

  [nkx,nky] = size(k_hat);
  n=size(u,1);
  Ku = real(ifft2( ((fft2(u,nkx,nky)) .* k_hat)));
  if nargin == 4
    Ku = Ku(1:nux,1:nuy);
  end
   end


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
function [w] = KKalphaI(nx,x,k_hat,L,alpha)

x_mat = reshape(x,nx,nx);
y_mat = integral_cgm(x_mat,k_hat,nx,nx);
w1_mat = integral_cgm(y_mat,conj(k_hat),nx,nx);
% L = speye(nx^2);
  w = w1_mat(:) + alpha*L*x ; % 
end
function [S] = PALM(nx,x,c_hat,L,alpha,gama)
x_mat = reshape(x,nx,nx);
y_mat = integral_cgm(x_mat,c_hat,nx,nx);
w1_mat = integral_cgm(y_mat,conj(c_hat),nx,nx);
 LI = speye(nx^2);
 
T1 = gama*(alpha*w1_mat(:) + gama*LI*x+ (alpha)*diag(diag(L))*x) ;%gamma*K^*K+alpha*I
 S1=T1;
 S=S1(:);
end

function [P] = PRE(nx,x,c_hat,A,B,C,D,alpha,gama)
if gama>=0.9 && gama <=1.2
    gama=1e-4;
else
    gama=1e-3;
end
BD = B'*inv(D) ;
L1 = 4*gama^2*BD*B;
L2 = 2*gama*BD*C*inv(D)*B;
L3 = 2*gama*A*BD*B;
L = (L1)^2 + L2 + L3  ;
J1=alpha*L2/2;
J2= L1/(4*gama);
J=J1+alpha*A*J2-2*J2^2+2*alpha*gama*J2;
x_mat = reshape(x,nx,nx);
y_mat = integral_cgm(x_mat,c_hat,nx,nx);
w1_mat = integral_cgm(y_mat,conj(c_hat),nx,nx);
 LI = speye(nx^2);
 
T1 = (alpha*w1_mat(:) + alpha*(LI-L-J)*x) ;%gamma*K^*K+alpha*LTV
 P1=T1;
 P=P1(:);
end

function p = ppsnr(x,y)

d = mean( mean( (x(:)-y(:)).^2 ) );
m1 = max( abs(x(:)) );
m2 = max( abs(y(:)) );
m = max(m1,m2);

p = 10*log10( m^2/d );
end 













