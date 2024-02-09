size(200);
import graph3;
import three;
// size(4cm,0);
currentprojection=perspective(6,6,4);
pen uppen = deepblue;
pen downpen = deepred;
triple circleCenter = (0,0,0);
path3 mycircle = circle(c=circleCenter, r=1,normal=Z);
draw(mycircle, black);


int n = 16;
triple p;

for(int i=0;i<n;++i){
    p =(cos(i*2*pi/n),sin(i*2*pi/n),0);
    dot(p);
  	
    if(unitrand()>0.5){
    	path3 n = p -- p + (0,0,0.3);
      draw(n, uppen, Arrow3(emissive(deepblue)));
    }
  else{
      path3 n = p -- p + (0,0,-0.3);
      draw(n, downpen, Arrow3(emissive(deepred)));
    }  
  
  }
