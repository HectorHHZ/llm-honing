// Restore legacy NanoJev prompts while retaining PR121 compiled branching.
// Keep the public game's /health + /decision contract. Model and branch logic
// come from the unmodified PR121 release. Bind only to loopback.
import http from 'node:http';
import {readFileSync} from 'node:fs';
import {RuntimeMetrics} from './runtime-metrics.mjs';
const release=process.env.HONING_RELEASE;
const {validateRequest, healthFor, decideWithHoning}=await import('./compat-bridge.mjs');
const questions=JSON.parse(readFileSync(new URL('./legacy-questions.json',import.meta.url)));
const mappings=JSON.parse(process.env.WORKERS_JSON);
for(const {port,upstream} of mappings){
 const metrics=new RuntimeMetrics();
 const backend={url:upstream,transport:'honing',label:'NanoJev'};
 http.createServer(async(req,res)=>{
  const send=(status,body)=>{if(!res.destroyed){res.writeHead(status,{'Content-Type':'application/json'});res.end(JSON.stringify(body));}};
  let started=null;
  try{
   if(req.method==='GET'&&req.url==='/metrics')return send(200,metrics.snapshot());
   if(req.method==='GET'&&req.url==='/health')return send(200,await healthFor(backend,'classic',AbortSignal.timeout(3000)));
   if(req.method!=='POST'||req.url!=='/decision')return send(404,{error:'Unknown route'});
   const chunks=[];let size=0;
   for await(const chunk of req){size+=chunk.length;if(size>1024*1024)return send(413,{error:'Request too large'});chunks.push(chunk);}
   let body;try{body=JSON.parse(Buffer.concat(chunks));}catch{return send(400,{error:'Invalid JSON'});}
   validateRequest(body,'classic');
   started=performance.now();metrics.inflight++;
   const result=await decideWithHoning(backend,'classic',body,AbortSignal.timeout(10000),{questions,legacyNanoJev:true});
   metrics.record({ok:true,status:200,ms:performance.now()-started,fire:result.controls.fire,permitted:result.constraints?.fire_permitted});
   send(200,result);
  }catch(error){if(started!==null)metrics.record({ok:false,status:error.status||502,ms:performance.now()-started});console.error(new Date().toISOString(),port,error.message);send(error.status||502,{error:error.message});}finally{if(started!==null)metrics.inflight--;}
 }).listen(port,'127.0.0.1',()=>console.log(`ready 127.0.0.1:${port} -> ${upstream}`));
}
