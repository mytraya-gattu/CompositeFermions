#!/usr/bin/env wolframscript
Nn=4; others[i_]:=Complement[Range[Nn],{i}];
espFrac[i_,n_,jk_]:=Coefficient[Normal[Series[Product[(1+t/(z[i]-z[j]))^jk,{j,others[i]}],{t,0,n}]],t,n];
compact[i_,n_,m_,jk_]:=z[i]^(n+m)*espFrac[i,n,jk];
Ji[i_,jk_]:=Product[(z[i]-z[j]),{j,others[i]}]^jk;
true[i_,n_,m_,jk_]:=D[z[i]^(n+m)*Ji[i,jk],{z[i],n}]/Ji[i,jk];
detOf[orb_,cfg_,jk_]:=Det[Table[orb[i,cfg[[a,1]],cfg[[a,2]],jk],{i,Nn},{a,Length[cfg]}]];
cfg={{0,0},{0,1},{1,0},{2,0}};  (* C3, aufbau *)
Do[
 Module[{r=Together[detOf[compact,cfg,jk]/detOf[true,cfg,jk]]},
  Print["jk=",jk,"  compact/true = ",If[FreeQ[Simplify[r],z],Row[{"CONST=",Simplify[r]}],"z-DEPENDENT"]]],
 {jk,{3/2,5/2,1}}];
