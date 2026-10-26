// Bounded, anonymous counters. No state, session token, nickname, or IP is retained.
export class RuntimeMetrics {
 constructor(){this.rows=[];this.started=Date.now();this.total=0;this.failed=0;this.inflight=0;}
 record(row){this.total++;if(!row.ok)this.failed++;this.rows.push({...row,at:Date.now()});if(this.rows.length>4096)this.rows.shift();}
 snapshot(){const rows=this.rows.filter(r=>r.at>Date.now()-300000),ok=rows.filter(r=>r.ok),times=ok.map(r=>r.ms).sort((a,b)=>a-b);const q=p=>times.length?Math.round(times[Math.ceil(times.length*p)-1]):null;
 return {at:new Date().toISOString(),started_at:new Date(this.started).toISOString(),total:this.total,failed_total:this.failed,inflight:this.inflight,window_seconds:300,samples:rows.length,errors:rows.length-ok.length,p50_ms:q(.5),p95_ms:q(.95),fire:ok.filter(r=>r.fire===true).length,fire_permitted:ok.filter(r=>r.permitted===true).length,status_counts:Object.fromEntries([...new Set(rows.map(r=>r.status))].map(s=>[s,rows.filter(r=>r.status===s).length]))};}
}
