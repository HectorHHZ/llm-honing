// Run after: npm install --prefix integrations/overcooked/.cache/browser playwright
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {spawn} = require('node:child_process');
const {chromium} = require('../.cache/browser/node_modules/playwright');
const root = path.resolve(__dirname, '..');
const python = path.join(root, '.venv', process.platform === 'win32' ? 'python.exe' : 'bin/python');
const executable = process.env.OVERCOOKED_BROWSER || (process.platform === 'win32' ?
    'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe' : undefined);
const modelEndpoint = process.env.OVERCOOKED_DECISION_SERVER;
const server = spawn(python, ['-m','integrations.overcooked.ui','--port','0',
    ...(modelEndpoint ? ['--decision-server', modelEndpoint] : [])], {
    cwd:path.resolve(root,'../..'), windowsHide:true, stdio:['ignore','pipe','pipe'],
});
let browser;
let replayServer;
let errors=[];
const output=path.join(root,'.cache','browser-qa');
fs.mkdirSync(output,{recursive:true});
async function main(){
    const url=await new Promise((resolve,reject)=>{
        const timeout=setTimeout(()=>reject(Error('GUI server did not start')),30000);
        server.stdout.on('data',chunk=>{const match=chunk.toString().match(/http:\/\/127\.0\.0\.1:\d+\//);if(match){clearTimeout(timeout);resolve(match[0]);}});
        server.once('exit',code=>{clearTimeout(timeout);reject(Error('GUI server exited '+code));});
    });
    browser=await chromium.launch({headless:true, executablePath:executable});
    const page=await browser.newPage({viewport:{width:1440,height:1000}});
    page.on('pageerror',error=>errors.push(error.message));
    await page.goto(url);
    await page.waitForFunction(()=>document.getElementById('step').textContent==='0');
    await page.waitForFunction(()=>document.getElementById('game').naturalWidth>0);
    await page.screenshot({path:path.join(output,'gui-initial.png'),fullPage:true});
    await page.click('#once');
    await page.waitForFunction(()=>document.getElementById('step').textContent==='1');
    if(modelEndpoint){
        await page.selectOption('#policy','model');
        for(let tick=2;tick<=4;tick++){
            await page.click('#once');
            await page.waitForFunction(t=>Number(document.getElementById('step').textContent)===t,tick);
        }
        await page.screenshot({path:path.join(output,'gui-model.png'),fullPage:true});
        const exported=await (await page.request.post(url+'api/export',{data:{}})).json();
        const rows=fs.readFileSync(path.join(exported.out,'decisions.jsonl'),'utf8').trim().split('\n').map(JSON.parse);
        assert.equal(rows.filter(r=>r.model_request && r.model_response).length,3);
        for(const row of rows.filter(r=>r.model_response)){
            assert.equal(row.chosen,row.model_response.answers.action.choice);
            assert.equal(row.model_request.id,row.model_response.id);
        }
        await page.selectOption('#policy','baseline');
    }
    await page.click('[data-tab=state]');
    assert.match(await page.locator('#json').innerText(),/players/);
    await page.click('[data-tab=map]');
    assert.match(await page.locator('#json').innerText(),/terrain/);
    await page.click('[data-tab=rules]');
    await page.click('#play');
    await page.waitForFunction(()=>Number(document.getElementById('soups').textContent)>=1,{timeout:30000});
    await page.click('#play');
    await page.waitForTimeout(400);
    const frozen=await page.locator('#step').innerText();
    await page.waitForTimeout(500);
    assert.equal(await page.locator('#step').innerText(),frozen,'pause must stop advancing');
    await page.screenshot({path:path.join(output,'gui-delivery.png'),fullPage:true});
    await page.click('#export');
    await page.waitForFunction(()=>document.getElementById('message').textContent.includes('export-'));
    const saved=(await page.locator('#message').innerText()).replace('Recording saved: ', '');
    assert.ok(fs.existsSync(path.join(saved,'steps.jsonl')));
    await page.selectOption('#dynamics','true');
    await page.click('#reset');
    await page.waitForFunction(()=>document.getElementById('step').textContent==='0');
    assert.equal((await (await page.request.get(url+'api/rules')).json()).old_dynamics,true);
    await page.selectOption('#policy','human');
    await page.locator('#game').click();
    await page.keyboard.press('w');
    await page.waitForFunction(()=>document.getElementById('step').textContent==='1');
    await page.screenshot({path:path.join(output,'gui-desktop.png'),fullPage:true});
    await page.selectOption('#layout','coordination_ring');
    await page.click('#reset');
    await page.waitForFunction(()=>document.getElementById('step').textContent==='0');
    assert.equal((await (await page.request.get(url+'api/map')).json()).layout,'coordination_ring');
    await page.setViewportSize({width:390,height:844});
    await page.screenshot({path:path.join(output,'gui-mobile.png'),fullPage:true});
    assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth),'mobile page must not overflow horizontally');
    replayServer=spawn(python,['-m','integrations.overcooked.ui','--port','0','--replay',saved],{
        cwd:path.resolve(root,'../..'),windowsHide:true,stdio:['ignore','pipe','pipe'],
    });
    const replayUrl=await new Promise((resolve,reject)=>{
        const timeout=setTimeout(()=>reject(Error('Replay server did not start')),30000);
        replayServer.stdout.on('data',chunk=>{const match=chunk.toString().match(/http:\/\/127\.0\.0\.1:\d+\//);if(match){clearTimeout(timeout);resolve(match[0]);}});
        replayServer.once('exit',code=>{clearTimeout(timeout);reject(Error('Replay server exited '+code));});
    });
    await page.setViewportSize({width:1440,height:1000});
    await page.goto(replayUrl);
    await page.waitForFunction(()=>document.getElementById('status').textContent.includes('replay'));
    await page.locator('#seek').evaluate(element=>{element.value='5';element.dispatchEvent(new Event('change'));});
    await page.waitForFunction(()=>document.getElementById('step').textContent==='5');
    await page.click('#reset');
    await page.waitForFunction(()=>document.getElementById('step').textContent==='0');
    assert.deepEqual(errors,[]);
    const result={passed:true,browser:'Microsoft Edge headless',checks:['native PNG','single step','autoplay delivery',
        'pause','state/map/rules panels','export','dynamics reset','keyboard input','layout switch','mobile layout',
        'verified replay seek','replay reset'],
        exported_run:saved,screenshots:output,model_gui_decisions:modelEndpoint?3:0};
    fs.writeFileSync(path.join(output,'result.json'),JSON.stringify(result,null,2));
    console.log(JSON.stringify(result));
}
main().catch(error=>{console.error(error);process.exitCode=1;}).finally(async()=>{if(browser)await browser.close();server.kill();if(replayServer)replayServer.kill();});
