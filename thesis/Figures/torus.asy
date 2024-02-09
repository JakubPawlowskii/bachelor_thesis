size(200);
import graph3;

pen surfPen=rgb(1,0.7,0)+opacity(0.5);
pen xarcPen=black+0.7bp;
pen yarcPen=black+0.7bp;
pen uppen = deepblue;
pen downpen = deepred;
currentprojection=perspective(5,4,6);

real R=2;
real a=1;

triple fs(pair t) {
  return ((R+a*Cos(t.y))*Cos(t.x),(R+a*Cos(t.y))*Sin(t.x),a*Sin(t.y));
}

real e = 0.001; // *this... is... infinitesimal!*
triple U(pair t) { return (fs((e,0)+t)-fs(t))/e; }
triple V(pair t) { return (fs((0,e)+t)-fs(t))/e; }

triple N(pair uv) {
  return 0.5*cross(U(uv),V(uv))/length(cross(U(uv),V(uv)));
}



surface s=surface(fs,(0,0),(360,360),8,8,Spline);
draw(s,surfPen,render(compression=Low,merge=true));

int m=14;
int n=7;
real arcFactor=1;
//triple adj = (0,1.4,0.7);
pair p,q,v;

for(int i=1;i<=n;++i){
  for(int j=0;j<m;++j){
    p=(j*360/m,(i%n)*360/n);
    q=(((j+arcFactor)%m)*360/m,i*360/n);
    v=(((j+arcFactor/2)%m)*360/m,i*360/n);
    draw(fs(p)..fs(v)..fs(q),xarcPen);
    q=(j*360/m,((i%n)-arcFactor)*360/n);
    draw(fs(p)..fs((p+q)/2)..fs(q),yarcPen);
    dot(fs(p));
   
    
    if(unitrand()>0.5){
    	path3 n = fs(p) -- shift(N(p))*fs(p);
      draw(n, uppen, Arrow3(emissive(deepblue)));
    }else{
      path3 n = fs(p) -- shift(-N(p))*fs(p);
      draw(n, downpen, Arrow3(emissive(deepred)));
    }
    //else{
  //  draw(n, Arrow3(emissive(blue)))
   // }
    
  }
}