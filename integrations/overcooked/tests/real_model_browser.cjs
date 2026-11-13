// Optional live acceptance, after evaluate_pilot has completed. No protocol stand-in.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {spawn, spawnSync} = require('node:child_process');
const {chromium} = require('../.cache/browser/node_modules/playwright');
const root = path.resolve(__dirname, '..');
const cwd = path.resolve(root, '../..');
const python = path.join(root, '.venv', 'python.exe');
const output = path.join(root, '.cache', 'real-model-browser');
const completed = JSON.parse(fs.readFileSync(path.join(root, 'runs/pilot-v1/completed.json'), 'utf8'));
const endpoint = completed.endpoint;
const headers = process.env.HONING_TOKEN ? {Authorization:'Bearer '+process.env.HONING_TOKEN} : {};
let browser, server;
async function health() {
    const response = await fetch(endpoint+'/v1/health', {headers});
    assert.ok(response.ok);
    const value = await response.json();
    assert.equal(value.ready, true);
    assert.equal(value.metadata.family, 'nanojev');
    assert.equal(value.metadata.weightsSha256, completed.selection.selected_checksum);
    return value;
}
async function main() {
    const model = await health();
    fs.mkdirSync(output, {recursive:true});
    server = spawn(python, ['-m','integrations.overcooked.ui','--port','0','--decision-server',endpoint],
        {cwd, windowsHide:true, stdio:['ignore','pipe','pipe']});
    const url = await new Promise((resolve,reject)=>{
        const timeout=setTimeout(()=>reject(Error('GUI startup timeout')),30000);
        server.stdout.on('data',chunk=>{const match=chunk.toString().match(/http:\/\/127\.0\.0\.1:\d+\//);
            if(match){clearTimeout(timeout);resolve(match[0]);}});
        server.once('exit',code=>{clearTimeout(timeout);reject(Error('GUI exited '+code));});
    });
    browser = await chromium.launch({headless:true, executablePath:process.env.OVERCOOKED_BROWSER ||
        'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
    const page = await browser.newPage({viewport:{width:1440,height:1000}});
    const errors=[];
    page.on('pageerror',error=>errors.push(error.message));
    await page.goto(url);
    await page.waitForFunction(()=>document.getElementById('step').textContent==='0');
    await page.selectOption('#policy','model');
    for(let step=1;step<=5;step++) {
        await page.click('#once');
        await page.waitForFunction(value=>document.getElementById('step').textContent===String(value),step,
            {timeout:30000});
    }
    await page.waitForFunction(()=>document.getElementById('game').complete &&
        document.getElementById('game').naturalWidth>0);
    await page.screenshot({path:path.join(output,'gui-model.png'),fullPage:true});
    const state=await (await page.request.get(url+'api/state')).json();
    assert.equal(state.frame.step,5);
    assert.equal(state.scenario.controlled_player,0);
    const saved=await (await page.request.post(url+'api/export',{data:{}})).json();
    const manifest=JSON.parse(fs.readFileSync(path.join(saved.out,'manifest.json'),'utf8'));
    assert.equal(manifest.kind,'interactive');
    const decisions=fs.readFileSync(path.join(saved.out,'decisions.jsonl'),'utf8').trim().split('\n').map(JSON.parse);
    assert.equal(decisions.length,5);
    assert.ok(decisions.every(row=>row.controller==='model' && row.model_request && row.model_response &&
        row.chosen===row.model_response.answers.action.choice));
    const verification=spawnSync(python,['-m','integrations.overcooked.verify',saved.out],{cwd,encoding:'utf8'});
    assert.equal(verification.status,0,verification.stderr);
    await health();
    assert.deepEqual(errors,[]);
    const result={passed:true,model:model.metadata,steps:5,controlled_player:0,partner:state.scenario.partner,
        actions:decisions.map(row=>row.chosen),exported_run:saved.out,native_replay:true,
        screenshot:path.join(output,'gui-model.png'),kind:'interactive GUI acceptance, not a graded episode'};
    fs.writeFileSync(path.join(output,'result.json'),JSON.stringify(result,null,2));
    console.log(JSON.stringify(result));
}
main().catch(error=>{console.error(error);process.exitCode=1;}).finally(async()=>{
    if(browser)await browser.close();
    if(server)server.kill();
});
