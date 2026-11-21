/** Create the filterable comparison workbook from verified local evidence. */
import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const [root, output, previews] = process.argv.slice(2);
if (!root || !output || !previews) throw Error("Expected project root, XLSX path and preview directory");
const read = async (...p) => JSON.parse(await fs.readFile(path.join(root, ...p), "utf8"));
const base = "reports/capture-full-quality/2026-11-21";
const frozen = "reports/capture-full-metric3d-30k/20261120-v1";
const n = await read(base,"numbers.json");
const d = await read(base,"detailed_numbers.json");
if (n.result_commit !== d.result_commit) throw Error("Different frozen result identities");
const wb = Workbook.create();
const tabs = ["指标总览","几何结果","距离分层","时间窗","局部与训练","逐帧配对"];
const [overview,surface,bands,windows,local,frames] = tabs.map(x => wb.worksheets.add(x));
const font="Microsoft YaHei", ink="#243B4B", header="#344D60", accent="#136F6C";
const fmt = {num:"0.0000",integer:"#,##0",pct:"0.00%",diff:"+0.0000;-0.0000;0.0000",
             pp:'+0.00" pp";-0.00" pp";0.00" pp"'};
const splitName = {val:"验证",test:"测试"};
const lab = {model:"模型深度",raw_pcd:"原始点云",fused_pcd:"支持融合点云",raw_mesh:"原始网格",
             pre_crop_mesh:"支持裁前网格",supported_mesh:"支持裁后网格",common:"两组支持网格共同命中"};
const col = (i) => { let s=""; while (i>0) {i--;s=String.fromCharCode(65+i%26)+s;i=Math.floor(i/26);} return s; };
const addr=(c,r)=>col(c)+r;
for (const sh of [overview,surface,bands,windows,local,frames]) sh.showGridLines=false;
overview.tabColor=accent;

function line(sh,row,last,value,size=10) {
  const r=sh.getRange("A"+row+":"+col(last)+row);
  r.merge();sh.getCell(row-1,0).values=[[value]];
  r.format.font={name:font,size,bold:size>12,color:size>12?ink:"#4A5B65"};
  r.format.rowHeight=size>12?27:22;
  r.format.verticalAlignment="center";
}
function top(sh,title,note,span=8) {
  line(sh,2,span,title,15);
  sh.getRange("A2:"+col(span)+"2").format.borders={bottom:{style:"thin",color:"#6B9F9B"}};
  line(sh,3,span,note);
}
function formatHeader(sh,row,last) {
  const h=sh.getRange("A"+row+":"+col(last)+row);
  h.format.fill=header;h.format.font={name:font,size:10,bold:true,color:"#FFFFFF"};
  h.format.horizontalAlignment="center";h.format.verticalAlignment="center";h.format.wrapText=true;h.format.rowHeight=32;
}
function block(sh,start,headers,records,widths,tableName) {
  const last=start+records.length, width=headers.length;
  sh.getRangeByIndexes(start-1,0,records.length+1,width).values=[headers,...records];
  const all=sh.getRange("A"+start+":"+col(width)+last);
  all.format.font={name:font,size:10,color:ink};all.format.verticalAlignment="center";all.format.rowHeight=20;
  if(tableName) {
    const t=sh.tables.add("A"+start+":"+col(width)+last,true,tableName);
    t.style="TableStyleLight9";t.showFilterButton=true;
  }
  for(let c=0;c<width;c++) sh.getRangeByIndexes(start-1,c,1,1).format.columnWidthPx=widths[c]??112;
  formatHeader(sh,start,width);
  return {first:start+1,last,width};
}
function numCol(sh,area,column,type) {
  sh.getRange(col(column)+area.first+":"+col(column)+area.last).format.numberFormat=type;
}
function formulas(sh,area,column,make) {
  sh.getRange(col(column)+area.first+":"+col(column)+area.last).formulas=
    Array.from({length:area.last-area.first+1},(_,i)=>[make(i+area.first)]);
}
const groupSrc="冻结结果 7d4a8a1。B0 为 Omnidata，B1 为 Metric3D v2。正负差值均按 B1 减 B0。";
const metrics=[
  ["PSNR (dB)","psnr","rgb"],["SSIM","ssim","rgb"],["LPIPS","lpips","rgb"],
  ["深度 MAE (m)","MAE_m","depth_full_reference"],["深度 RMSE (m)","RMSE_m","depth_full_reference"],
  ["AbsRel","AbsRel","depth_full_reference"],["深度 P50 (m)","absolute_error_P50_m","depth_full_reference"],
  ["深度 P95 (m)","absolute_error_P95_m","depth_full_reference"]];
const fh=["划分","源帧","采集时间 (s)","参考像素"];
for(const [name] of metrics) fh.push("B0 "+name,"B1 "+name,"差值 "+name);
fh.push("原 s01 PSNR (dB)","原 s01 深度 MAE (m)","PSNR 与深度 MAE 配对方向");
const pairRows=[];
for(const split of ["val","test"]) {
  const p=await read(frozen,"paired_"+split+".json");
  const old=new Map(p.s01_same_sources.map(r=>[r.id,r]));
  for(const r of p.frames) {
    const row=[splitName[split],r.id,r.relative_s,r.reference_pixels];
    for(const [,key,section] of metrics) row.push(r.B0[section][key],r.B1[section][key],null);
    const o=old.get(r.id);row.push(o?.s01.psnr??null,o?.s01_depth?.MAE_m??null,null);
    pairRows.push(row);
  }
}
top(frames,"逐帧配对","两组在相同源帧上的数值。来源为冻结报告 paired_val.json 和 paired_test.json。",10);
const fp=block(frames,5,fh,pairRows,[78,136,124,125,...fh.slice(4).map(()=>146)],"PairedFrames");
numCol(frames,fp,3,"0.000");numCol(frames,fp,4,fmt.integer);
for(let i=0;i<metrics.length;i++) {
  const c=5+3*i,percent=metrics[i][1]==="AbsRel";
  numCol(frames,fp,c,percent?fmt.pct:fmt.num);numCol(frames,fp,c+1,percent?fmt.pct:fmt.num);
  numCol(frames,fp,c+2,percent?fmt.pct:fmt.diff);
  formulas(frames,fp,c+2,r=>"="+addr(c+1,r)+"-"+addr(c,r));
}
numCol(frames,fp,29,fmt.num);numCol(frames,fp,30,fmt.num);
formulas(frames,fp,31,r=>'=IF(AND(G'+r+'>0,P'+r+'<0),"两项改善",IF(G'+r+'>0,"仅 PSNR 改善",IF(P'+r+'<0,"仅深度 MAE 改善","两项均未改善")))');
frames.getRange("AE5").format.columnWidthPx=205;
frames.freezePanes.freezeRows(5);
frames.freezePanes.freezeColumns(2);

top(surface,"网格、点云与模型深度","正确覆盖的分母为全部有效传感器参考像素。MAE、RMSE、P50、P95 和有符号均值取命中处。",10);
const shdr=["划分","模型","结果层","参考像素","命中像素","未命中","≤5 cm 像素","≤10 cm 像素","≤20 cm 像素","自适应正确像素","前方冲突像素",
  "命中率","≤5 cm 正确覆盖","≤10 cm 正确覆盖","≤20 cm 正确覆盖","自适应正确覆盖","前方冲突比例","命中 MAE (m)","命中 RMSE (m)",
  "命中 P50 (m)","命中 P95 (m)","有符号均值 (m)","模型 AbsRel"];
const sr=[];
for(const split of ["val","test"]) for(const label of ["B0","B1"]) for(const [kind,show] of Object.entries(lab)) {
  const key=kind==="common"?"common_supported_models__"+label+"_supported_mesh":label+"_"+kind;
  const v=n.splits[split].surfaces[key];
  sr.push([splitName[split],label,show,v.reference,v.hits,null,v.correct_05,v.correct_10,v.correct_20,v.correct_tau,v.foreground,
    null,null,null,null,null,null,v.MAE_hit_m,v.RMSE_hit_m,v.P50_hit_m,v.P95_hit_m,v.signed_mean_hit_m,
    kind==="model"?n.splits[split].models[label].depth.AbsRel:null]);
}
const sp=block(surface,5,shdr,sr,[78,78,210,...shdr.slice(3).map(()=>135)],"GeometryMetrics");
for(let c=4;c<=11;c++)numCol(surface,sp,c,fmt.integer);
formulas(surface,sp,6,r=>"=D"+r+"-E"+r);
for(const [c,input] of [[12,"E"],[13,"G"],[14,"H"],[15,"I"],[16,"J"],[17,"K"]]) {
  formulas(surface,sp,c,r=>"="+input+r+"/D"+r);numCol(surface,sp,c,fmt.pct);
}
for(let c=18;c<=22;c++)numCol(surface,sp,c,fmt.num);
numCol(surface,sp,23,fmt.pct);surface.freezePanes.freezeRows(5);surface.freezePanes.freezeColumns(3);

top(bands,"按传感器参考距离分层","固定距离分组。最后一组包含 6 m，各区间均按有效参考像素合并。负偏差表示预测比参考更近。",11);
const bh=["划分","距离区间 (m)","参考像素","参考占比","B0 MAE (m)","B1 MAE (m)","MAE 差 (m)","B0 RMSE (m)","B1 RMSE (m)",
 "B0 P95 (m)","B1 P95 (m)","B0 10 cm 覆盖","B1 10 cm 覆盖","B0 偏差 (m)","B1 偏差 (m)","B0 >1 m 比例","B1 >1 m 比例","占净误差减少比例","B0 AbsRel","B1 AbsRel","B0 P50 (m)","B1 P50 (m)","B0 P99 (m)","B1 P99 (m)"];
const br=[];
for(const split of ["val","test"]) for(const r of d.splits[split].depth_bands) {
  const a=r.B0,b=r.B1;
  br.push([splitName[split],"["+r.low_m+", "+r.high_m+(r.high_inclusive?"]":")"),r.reference,r.share,a.MAE_m,b.MAE_m,null,
           a.RMSE_m,b.RMSE_m,a.P95_m,b.P95_m,a.coverage_10,b.coverage_10,a.signed_bias_m,b.signed_bias_m,
           a.fraction_over_1m,b.fraction_over_1m,r.share_of_net_absolute_error_saved,a.AbsRel,b.AbsRel,a.P50_m,b.P50_m,a.P99_m,b.P99_m]);
}
const bp=block(bands,5,bh,br,[78,140,...bh.slice(2).map(()=>136)],"DepthBands");
numCol(bands,bp,3,fmt.integer);
for(let c=4;c<=bh.length;c++)numCol(bands,bp,c,[4,12,13,16,17,18,19,20].includes(c)?fmt.pct:fmt.num);
formulas(bands,bp,7,r=>"=F"+r+"-E"+r);numCol(bands,bp,7,fmt.diff);bands.freezePanes.freezeRows(5);

top(windows,"30 秒时间窗与留一窗复查","最后一个窗口只有 5 帧。留一窗列使用剔除本行窗口后的全部剩余帧，MAE 按参考像素合并。",10);
const wh=["划分","窗起点 (s)","窗上界 (s)","帧数","参考像素","B0 PSNR (dB)","B1 PSNR (dB)","PSNR 差 (dB)",
 "B0 MAE (m)","B1 MAE (m)","MAE 差 (m)","净误差减少占比","去该窗 PSNR 差","去该窗 MAE 降幅"];
const wr=[];
for(const split of ["val","test"]) {
  const ws=n.splits[split].windows;
  const net=ws.reduce((sum,x)=>sum+x.absolute_error_sum_saved_m,0);
  for(let i=0;i<ws.length;i++) {
    const x=ws[i],l=d.splits[split].leave_one_window_out[i];
    wr.push([splitName[split],x.start_s,x.end_exclusive_s,x.frames,x.reference,x.B0.PSNR_dB,x.B1.PSNR_dB,null,x.B0.MAE_m,x.B1.MAE_m,
             null,x.absolute_error_sum_saved_m/net,l.PSNR_B1_minus_B0_dB,l.MAE_relative_drop]);
  }
}
const wp=block(windows,5,wh,wr,[78,105,105,77,130,...wh.slice(5).map(()=>143)],"TimeWindows");
numCol(windows,wp,5,fmt.integer);
for(let c=6;c<=wh.length;c++) numCol(windows,wp,c,[12,14].includes(c)?fmt.pct:fmt.num);
formulas(windows,wp,8,r=>"=G"+r+"-F"+r);formulas(windows,wp,11,r=>"=J"+r+"-I"+r);
numCol(windows,wp,8,fmt.diff);numCol(windows,wp,11,fmt.diff);windows.freezePanes.freezeRows(5);

top(local,"固定局部、同源帧与训练容量","局部选自三个训练视角。误差只比较两组最终支持网格共同命中的像素。",7);
const cropNames={table:"桌沿",pillar:"柱面",chair_legs:"椅腿"};
const lr=Object.entries(n.fixed_training_crops).map(([k,x])=>[cropNames[k],x.train_frame,x.reference,x.common_hits,x.B0_MAE_m,x.B1_MAE_m,null]);
const lp=block(local,5,["局部","训练帧","参考像素","共同命中","B0 MAE (m)","B1 MAE (m)","差值 (m)"],lr,[175,145,133,130,145,145,146],"FixedCrops");
for(const c of [3,4])numCol(local,lp,c,fmt.integer);for(const c of [5,6,7])numCol(local,lp,c,fmt.num);
formulas(local,lp,7,r=>"=F"+r+"-E"+r);
line(local,11,7,"训练 checkpoint 中读取的高斯数",12);
const trainingRows=n.gaussians.B0.map(([step,count],i)=>[step,count,n.gaussians.B1[i][1],null,null]);
const gp=block(local,13,["保存 step","B0 高斯数","B1 高斯数","高斯数差","B1 相对 B0"],trainingRows,[175,145,133,130,145],"GaussianCounts");
for(const c of [1,2,3,4])numCol(local,gp,c,fmt.integer);
formulas(local,gp,4,r=>"=C"+r+"-B"+r);formulas(local,gp,5,r=>"=C"+r+"/B"+r+"-1");numCol(local,gp,5,fmt.pct);
line(local,22,7,"旧 s01 同源留出帧的平均 PSNR",12);
const earlyRows=["val","test"].map(s=>{const e=n.splits[s].old_s01_24;return [splitName[s],24,e.s01,e.B0,e.B1,null,e.improved]});
const ep=block(local,24,["划分","相同源帧数","s01 (dB)","B0 (dB)","B1 (dB)","B1−B0 (dB)","PSNR 改善帧数"],earlyRows,[175,145,133,130,145,145,146],"SameSource");
for(const c of [3,4,5,6])numCol(local,ep,c,fmt.num);formulas(local,ep,6,r=>"=E"+r+"-D"+r);
line(local,29,7,"逐帧表的原 s01 列可筛出全部 48 个同源帧。",10);

top(overview,"capture-full 重建质量对照",groupSrc,8);
const oh=["指标","单位","验证 B0","验证 B1","验证差值","测试 B0","测试 B1","测试差值"];
const defs=[
 ["PSNR","dB","paired","E","F"],["SSIM","指数","paired","H","I"],["LPIPS","指数","paired","K","L"],
 ["模型深度 MAE","m","模型深度","R"],["模型深度 RMSE","m","模型深度","S"],["模型 AbsRel","%","模型深度","W"],
 ["模型深度 P50","m","模型深度","T"],["模型深度 P95","m","模型深度","U"],
 ["模型 10 cm 正确覆盖","%","模型深度","N"],["原始网格 10 cm 正确覆盖","%","原始网格","N"],
 ["支持网格命中率","%","支持裁后网格","L"],["支持网格 5 cm 正确覆盖","%","支持裁后网格","M"],
 ["支持网格 10 cm 正确覆盖","%","支持裁后网格","N"],["支持网格 20 cm 正确覆盖","%","支持裁后网格","O"],
 ["支持网格自适应正确覆盖","%","支持裁后网格","P"],["支持网格前方冲突","%","支持裁后网格","Q"],
 ["支持网格命中 MAE","m","支持裁后网格","R"],["支持网格命中 P95","m","支持裁后网格","U"]];
const op=block(overview,5,oh,defs.map(x=>[x[0],x[1],null,null,null,null,null,null]),[224,82,115,115,129,115,115,129],null);
const frameRange=x=>"'逐帧配对'!$"+x+"$"+fp.first+":$"+x+"$"+fp.last;
const surfaceRange=x=>"'几何结果'!$"+x+"$"+sp.first+":$"+x+"$"+sp.last;
defs.forEach(([name,unit,kind,b0,b1],i)=>{
  const row=op.first+i;
  for(const [split,ac,bc,dc] of [["验证",3,4,5],["测试",6,7,8]]) {
    for(const [label,c] of [["B0",ac],["B1",bc]]) {
      const formula=kind==="paired"?"=AVERAGEIFS("+frameRange(label==="B0"?b0:b1)+","+frameRange("A")+',"' +split+'")':
        "=SUMIFS("+surfaceRange(b0)+","+surfaceRange("A")+',"' +split+'",'+surfaceRange("B")+',"' +label+'",'+surfaceRange("C")+',"' +kind+'")';
      overview.getCell(row-1,c-1).formulas=[[formula]];
      overview.getCell(row-1,c-1).format.numberFormat=unit==="%"?fmt.pct:fmt.num;
    }
    overview.getCell(row-1,dc-1).formulas=[["=("+addr(bc,row)+"-"+addr(ac,row)+")"+(unit==="%"?"*100":"")]];
    overview.getCell(row-1,dc-1).format.numberFormat=unit==="%"?fmt.pp:fmt.diff;
  }
});
line(overview,26,8,"测试集 184 帧中，119 帧 PSNR 提高，113 帧深度 MAE 下降，79 帧同时改善。",10);
line(overview,27,8,"测试的全部有效传感器参考像素为 29,100,252 个，占 RGB 像素的 17.16%。未命中计入正确覆盖率分母。",10);
line(overview,28,8,"自适应容差取 5 cm 与参考深度的 3% 中的较大值。pp 表示百分点。",10);
line(overview,30,8,"来源及口径",12);
line(overview,31,8,"科学结果提交 "+n.result_commit+"。距离分层重读 H 盘中的原始传感器 Z 和两模型逐帧渲染 Z。",10);
line(overview,32,8,"验证用于后续调整。帧的平均分位数与合并像素分位数不同，表中总览 P50/P95 均为合并像素值。",10);
line(overview,33,8,"https://github.com/kaiwen123-yang/DN-Splatter/tree/"+n.result_commit+"/reports/capture-full-metric3d-30k/20261120-v1",10);

wb.recalculate();
const expect=(a,b,tol=2e-7)=>{if(typeof a!=="number"||Math.abs(a-b)>tol)throw Error("Workbook value mismatch "+a+" "+b)};
expect(overview.getRange("G6").values[0][0],n.splits.test.models.B1.rgb.psnr);
expect(overview.getRange("G9").values[0][0],n.splits.test.models.B1.depth.MAE_m);
expect(overview.getRange("G18").values[0][0],n.splits.test.surfaces.B1_supported_mesh.correct_10_pct_reference/100);
expect(frames.getRange("G6").values[0][0],pairRows[0][5]-pairRows[0][4]);
await fs.mkdir(previews,{recursive:true});await fs.mkdir(path.dirname(output),{recursive:true});
for(const [s,range,label] of [[overview,"A1:H33","overview"],[surface,"A1:K10","surface"],[bands,"A1:K13","bands"],
                             [windows,"A1:J12","windows"],[local,"A1:G29","local"],[frames,"A1:M11","frames"]]) {
  const image=await wb.render({sheetName:s.name,range,scale:1,format:"png"});
  await fs.writeFile(path.join(previews,label+".png"),new Uint8Array(await image.arrayBuffer()));
}
const inspection=await wb.inspect({kind:"region",sheetId:overview.name,range:"C6:H10",maxChars:1400,tableMaxRows:5,tableMaxCols:6});
await fs.writeFile(path.join(previews,"inspection.txt"),inspection.ndjson??String(inspection));
const xlsx=await SpreadsheetFile.exportXlsx(wb);await xlsx.save(output);
console.log("Exported",output,"paired frames",pairRows.length,"sheets",tabs.length);
