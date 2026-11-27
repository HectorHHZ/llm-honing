using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.UI;

// LOCAL screen-only adapter. Rendered label masks are instrumentation, never model pixels.
public static class VisibleObservation
{
    [Serializable] public class Frame { public int id, tick; public float simulation_time; }
    [Serializable] public class View { public int[] screen_rect; public float[] world_from_screen; public string world_units = "Unity world units; screen x right/y down"; }
    [Serializable] public class Legend { public string unknown = "? unknown/unsupported/partly transparent; never safe", background = ". visible background/no visible obstacle; not a collision guarantee", solid = "# visible solid-looking wall", platform = "= visible platform surface/body", hazard = "! visible spike/projectile hazard", actor = "@ visible player/enemy sprite", interactable = "+ visible interactable; no hidden identity"; }
    [Serializable] public class Terrain { public string status = "available", coverage = "full"; public int[] origin = {0,0}, cell_size; public string[] rows; public Legend legend = new Legend(); }
    [Serializable] public class Obj { public string id, kind; public int[] bounds; }
    [Serializable] public class Self { public string object_id; }
    [Serializable] public class PublicInt { public int value; public string evidence = "hud"; }
    [Serializable] public class Hud { public PublicInt health; }
    [Serializable] public class Disclosure { public string visibility, geometry, unsupported, resolution, debug; }
    [Serializable] public class Capabilities { public bool global_debug = true, pixels_to_model = false, hidden_own_state = false; }
    [Serializable] public class Envelope {
        public string schema = "honing.visible.v1", adapter = "hollow-knight-visible-v1", profile = "visible", game = "hollow-knight-imitation";
        public Frame frame; public View view; public Terrain terrain; public Obj[] objects; public Self self; public Hud hud;
        public string[] rules; public Capabilities capabilities = new Capabilities(); public Disclosure disclosure;
    }
    class Saved { public SpriteRenderer renderer; public Sprite sprite; public Color color; public Material material; public bool enabled; }
    static Dictionary<Texture2D,Texture2D> masks = new Dictionary<Texture2D,Texture2D>();
    static Dictionary<Sprite,Sprite> sprites = new Dictionary<Sprite,Sprite>();
    static Material material;
    const int Width = 960, Height = 540;
    static int ColorKey(Color32 c) { return c.r | c.g << 8 | c.b << 16; }
    static Color32 IdColor(int index) { return new Color32((byte)(30+(index%12)*18),(byte)(30+((index/12)%12)*18),(byte)(30+((index/144)%12)*18),255); }
    static Texture2D Mask(Texture2D original)
    {
        Texture2D result;
        if (masks.TryGetValue(original, out result)) return result;
        var target = RenderTexture.GetTemporary(original.width,original.height,0,RenderTextureFormat.ARGB32,RenderTextureReadWrite.Linear);
        var previous = RenderTexture.active;
        Graphics.Blit(original,target); RenderTexture.active = target;
        var readable = new Texture2D(original.width,original.height,TextureFormat.RGBA32,false,true);
        readable.ReadPixels(new Rect(0,0,original.width,original.height),0,0); readable.Apply();
        RenderTexture.active = previous; RenderTexture.ReleaseTemporary(target);
        var source = readable.GetPixels32(); var output = new Color32[source.Length];
        for(int y=0;y<original.height;y++) for(int x=0;x<original.width;x++) {
            bool any=false, all=true;
            // Conservative one-texel dilation/erosion covers original bilinear alpha edges.
            for(int dy=-1;dy<=1;dy++) for(int dx=-1;dx<=1;dx++) {
                int xx=Mathf.Clamp(x+dx,0,original.width-1), yy=Mathf.Clamp(y+dy,0,original.height-1);
                byte alpha=source[yy*original.width+xx].a; any |= alpha>0; all &= alpha==255;
            }
            output[y*original.width+x] = all ? new Color32(255,255,255,255) : any ? new Color32(0,0,0,255) : new Color32(0,0,0,0);
        }
        readable.SetPixels32(output); readable.filterMode=FilterMode.Point; readable.wrapMode=original.wrapMode; readable.Apply();
        masks[original]=readable; return readable;
    }
    static Sprite MaskSprite(Sprite original)
    {
        Sprite result;
        if(sprites.TryGetValue(original,out result)) return result;
        if(original.packed && original.packingRotation != SpritePackingRotation.None) return null;
        Rect rect=original.textureRect;
        Vector2 pivot=new Vector2((original.pivot.x-original.textureRectOffset.x)/rect.width,(original.pivot.y-original.textureRectOffset.y)/rect.height);
        result=Sprite.Create(Mask(original.texture),rect,pivot,original.pixelsPerUnit,0,SpriteMeshType.FullRect,original.border);
        result.name="semantic-mask"; sprites[original]=result; return result;
    }
    static char Kind(SpriteRenderer r, PlayerController player, out string kind)
    {
        if(player!=null && r.GetComponent<PlayerController>() == player) { kind="player"; return '@'; }
        if(player!=null && r.GetComponentInParent<PlayerController>() == player) { kind="visible_equipment"; return ':'; }
        if(r.GetComponentInParent<EnemyController>() != null) { kind="enemy"; return '@'; }
        string layer=LayerMask.LayerToName(r.gameObject.layer);
        if(layer=="Trap" || layer=="Projectile") { kind="visible_hazard"; return '!'; }
        if(layer=="Platform") { kind="platform"; return '='; }
        if(layer=="Wall" || layer=="AirWall") { kind="wall"; return '#'; }
        if(layer=="Switch" || r.GetComponent<NextLevel>() != null) { kind="interactable"; return '+'; }
        // Decorative/background colliders do not reveal collision semantics.
        kind="visible_scenery"; return ':';
    }
    static bool InCamera(Camera camera, Renderer r)
    {
        return r.enabled && r.gameObject.activeInHierarchy && (camera.cullingMask & (1<<r.gameObject.layer))!=0 && GeometryUtility.TestPlanesAABB(GeometryUtility.CalculateFrustumPlanes(camera),r.bounds);
    }
    static bool HasRenderedGraphic(Graphic graphic)
    {
        if(graphic==null || !graphic.isActiveAndEnabled || graphic.canvas==null || !graphic.canvas.enabled || graphic.color.a<=0 || graphic.canvasRenderer.cull) return false;
        var text=graphic as Text;if(text!=null && String.IsNullOrEmpty(text.text))return false;
        foreach(var group in graphic.GetComponentsInParent<CanvasGroup>())if(group.alpha<=0)return false;
        return true;
    }
    static Rect GraphicRect(Graphic graphic)
    {
        var corners=new Vector3[4];graphic.rectTransform.GetWorldCorners(corners);var canvas=graphic.canvas;
        var screen=corners.Select(c=>RectTransformUtility.WorldToScreenPoint(canvas.renderMode==RenderMode.ScreenSpaceOverlay?null:canvas.worldCamera,c)).ToArray();
        float lx=screen.Min(c=>c.x),rx=screen.Max(c=>c.x),by=screen.Min(c=>c.y),ty=screen.Max(c=>c.y);
        return new Rect(lx,Height-ty,rx-lx,ty-by);
    }
    static bool PublicGraphicVisible(Graphic graphic)
    {
        if(!HasRenderedGraphic(graphic) || graphic.color.a<1f || graphic.depth<0 || graphic.canvasRenderer.GetAlpha()<1f)return false;
        foreach(var group in graphic.GetComponentsInParent<CanvasGroup>())if(group.alpha<1f)return false;
        var canvas=graphic.canvas;
        if(canvas.renderMode!=RenderMode.ScreenSpaceOverlay && (canvas.renderMode!=RenderMode.ScreenSpaceCamera || canvas.worldCamera!=Camera.main))return false;
        var rect=GraphicRect(graphic);
        if(rect.width<=0 || rect.height<=0 || rect.xMin<0 || rect.yMin<0 || rect.xMax>Width || rect.yMax>Height)return false;
        foreach(var other in UnityEngine.Object.FindObjectsOfType<Graphic>()) {
            if(other==graphic || !HasRenderedGraphic(other))continue;
            if(other.canvas==canvas && other.depth<=graphic.depth)continue;
            // Cross-canvas order or transparent padding is conservatively treated
            // as a possible occluder; omit the HUD value rather than leak it.
            if(GraphicRect(other).Overlaps(rect))return false;
        }
        return true;
    }
    static bool HudPixel(int x,int y, bool capture)
    {
        // The bridge label is visible to the human in captured videos.
        return capture && x>=8 && x<668 && y>=8 && y<56;
    }
    public static Envelope Capture(PlayerController player,int id,int frame,bool capture,string session)
    {
        var camera=Camera.main;
        if(camera==null || !camera.orthographic || camera.transform.rotation!=Quaternion.identity || camera.pixelWidth!=Width || camera.pixelHeight!=Height)
            throw new InvalidOperationException("Visible adapter supports verified orthographic axis-aligned 960x540 camera only.");
        int cell=20; string cellText=Environment.GetEnvironmentVariable("HONING_VISIBLE_CELL_PIXELS");
        if(cellText!=null && (!Int32.TryParse(cellText,out cell) || cell<5 || cell>120 || Width%cell!=0 || Height%cell!=0)) throw new InvalidOperationException("Cell pixels must divide width and height, within 5..120.");
        var renderers=UnityEngine.Object.FindObjectsOfType<SpriteRenderer>().Where(r=>r.sprite!=null && InCamera(camera,r)).ToArray();
        var saved=new List<Saved>(); var kinds=new Dictionary<int,string>(); var chars=new Dictionary<int,char>(); var sourceIds=new Dictionary<int,string>();
        int selfKey=-1;
        if(UnityEngine.Object.FindObjectsOfType<Renderer>().Any(r=>!(r is SpriteRenderer) && InCamera(camera,r)))
            throw new InvalidOperationException("Unsupported visible non-sprite renderer; refusing visible snapshot.");
        var uiRects=new List<Rect>();
        foreach(var graphic in UnityEngine.Object.FindObjectsOfType<Graphic>()) {
            if(!HasRenderedGraphic(graphic)) continue;
            var canvas=graphic.canvas; if(canvas==null || !canvas.enabled)continue;
            var corners=new Vector3[4];graphic.rectTransform.GetWorldCorners(corners);
            var screen=corners.Select(c=>RectTransformUtility.WorldToScreenPoint(canvas.renderMode==RenderMode.ScreenSpaceOverlay?null:canvas.worldCamera,c)).ToArray();
            float lx=screen.Min(c=>c.x), rx=screen.Max(c=>c.x), by=screen.Min(c=>c.y), ty=screen.Max(c=>c.y);
            uiRects.Add(new Rect(lx,Height-ty,rx-lx,ty-by));
        }
        if(material==null) material=new Material(Shader.Find("Sprites/Default"));
        var oldTarget=camera.targetTexture; var oldActive=RenderTexture.active; var oldFlags=camera.clearFlags; var oldBackground=camera.backgroundColor; float oldAspect=camera.aspect;
        var target=new RenderTexture(Width,Height,24,RenderTextureFormat.ARGB32,RenderTextureReadWrite.Linear); target.filterMode=FilterMode.Point;
        var image=new Texture2D(Width,Height,TextureFormat.RGBA32,false,true);
        try {
            for(int i=0;i<renderers.Length;i++) {
                var r=renderers[i]; var prior=new Saved {renderer=r,sprite=r.sprite,color=r.color,material=r.sharedMaterial,enabled=r.enabled}; saved.Add(prior);
                var color=IdColor(i+1); int key=ColorKey(color); string kind; char glyph=Kind(r,player,out kind);
                kinds[key]=kind; chars[key]=glyph; sourceIds[key]="v"+Math.Abs(r.GetInstanceID());
                if(kind=="player") selfKey=key;
                bool supported=prior.material!=null && prior.material.shader.name=="Sprites/Default" && r.maskInteraction==SpriteMaskInteraction.None;
                if(!supported) throw new InvalidOperationException("Unsupported sprite shader/mask; refusing visible snapshot.");
                if(prior.color.a<=0) {r.enabled=false; continue;}
                Sprite mask=null; if(supported) { try {mask=MaskSprite(prior.sprite);} catch(Exception) {supported=false;} }
                // Unsupported and translucent renderers conservatively block semantics as unknown.
                if(mask!=null) r.sprite=mask;
                r.sharedMaterial=material;
                r.color=supported && mask!=null && prior.color.a>=1f ? (Color)color : new Color(0,0,0,1);
            }
            camera.targetTexture=target; camera.aspect=oldAspect; camera.clearFlags=CameraClearFlags.SolidColor; camera.backgroundColor=new Color32(2,3,4,255);
            camera.Render(); RenderTexture.active=target; image.ReadPixels(new Rect(0,0,Width,Height),0,0); image.Apply();
        } finally {
            camera.targetTexture=oldTarget; camera.clearFlags=oldFlags; camera.backgroundColor=oldBackground; camera.aspect=oldAspect; RenderTexture.active=oldActive;
            foreach(var s in saved) { s.renderer.sprite=s.sprite; s.renderer.color=s.color; s.renderer.sharedMaterial=s.material; s.renderer.enabled=s.enabled; }
            UnityEngine.Object.Destroy(target);
        }
        var pixels=image.GetPixels32(); var minX=new Dictionary<int,int>();var minY=new Dictionary<int,int>();var maxX=new Dictionary<int,int>();var maxY=new Dictionary<int,int>();
        int cols=Width/cell, rows=Height/cell; var counts=new Dictionary<char,int>[rows*cols];
        for(int i=0;i<counts.Length;i++) counts[i]=new Dictionary<char,int>();
        int backgroundKey=ColorKey(new Color32(2,3,4,255));
        for(int y=0;y<Height;y++) for(int x=0;x<Width;x++) {
            int key=ColorKey(pixels[(Height-1-y)*Width+x]); char glyph;
            if(HudPixel(x,y,capture) || uiRects.Any(rect=>rect.Contains(new Vector2(x,y)))) { glyph='?'; key=-1; }
            else if(key==backgroundKey) glyph='.';
            else if(!chars.TryGetValue(key,out glyph)) glyph='?';
            var counter=counts[(y/cell)*cols+x/cell]; if(!counter.ContainsKey(glyph))counter[glyph]=0;counter[glyph]++;
            if(key>=0 && kinds.ContainsKey(key)) {
                if(!minX.ContainsKey(key)){minX[key]=maxX[key]=x;minY[key]=maxY[key]=y;}
                else {minX[key]=Math.Min(minX[key],x);maxX[key]=Math.Max(maxX[key],x);minY[key]=Math.Min(minY[key],y);maxY[key]=Math.Max(maxY[key],y);}
            }
        }
        var grid=new string[rows];
        for(int y=0;y<rows;y++) {var row=new char[cols];for(int x=0;x<cols;x++) {
            var counter=counts[y*cols+x];int mask=0;char[] alphabet={'.','#','=','!','@','+','?',':'};
            for(int bit=0;bit<alphabet.Length;bit++)if(counter.ContainsKey(alphabet[bit]))mask|=1<<bit;
            row[x]=(char)(0x100+mask);
        }grid[y]=new string(row);}
        var objects=minX.Keys.OrderBy(k=>sourceIds[k]).Select(k=>new Obj{id=sourceIds[k],kind=kinds[k],bounds=new[]{minX[k],minY[k],maxX[k]-minX[k]+1,maxY[k]-minY[k]+1}}).ToArray();
        var bottomLeft=camera.ScreenToWorldPoint(new Vector3(0,Height,0)); var right=camera.ScreenToWorldPoint(new Vector3(1,Height,0));var down=camera.ScreenToWorldPoint(new Vector3(0,Height-1,0));
        var hud=new Hud();var health=UnityEngine.Object.FindObjectOfType<HealthDisplay>();
        if(health!=null && health.isActiveAndEnabled && health.GetComponentInParent<Canvas>()!=null && health.GetComponentInParent<Canvas>().enabled)
            {
            var hearts=new[]{health.heart1,health.heart2,health.heart3,health.heart4,health.heart5}.Select(go=>go==null?null:go.GetComponent<Image>()).ToArray();
            if(hearts.All(h=>h!=null && h.sprite!=null && (h.sprite==health.healthFull || h.sprite==health.healthEmpty) && PublicGraphicVisible(h))) hud.health=new PublicInt{value=hearts.Count(h=>h.sprite==health.healthFull)};
        }
        // Human-verification artifact only. It is never serialized as a model image.
        if(Environment.GetEnvironmentVariable("HONING_VISIBLE_MASK_CAPTURE")=="1") System.IO.File.WriteAllBytes(System.IO.Path.Combine(session,"semantic-"+id.ToString("D06")+".png"),image.EncodeToPNG());
        UnityEngine.Object.Destroy(image);
        return new Envelope {frame=new Frame{id=id,tick=frame,simulation_time=Time.time},
            view=new View{screen_rect=new[]{0,0,Width,Height},world_from_screen=new[]{right.x-bottomLeft.x,right.y-bottomLeft.y,down.x-bottomLeft.x,down.y-bottomLeft.y,bottomLeft.x,bottomLeft.y}},
            terrain=new Terrain{cell_size=new[]{cell,cell},rows=grid},objects=objects,self=new Self{object_id=minX.ContainsKey(selfKey)?sourceIds[selfKey]:null},hud=hud,
            rules=new[]{"Reach a visible level exit while staying alive. No direction to an offscreen exit is supplied.","Walk left/right; jump permits a second jump before landing. Walls permit wall climbing/jumping.","Dash and sword attacks obey native cooldowns; attack up/down directions are supported, downward sword attack requires airborne state.","Visible spikes/projectiles can hurt or kill. A gap/unknown cell is not proof of safety.","Screen x increases right and screen y increases down. Each action holds for the requested render-frame window; the game pauses during model inference."},
            disclosure=new Disclosure{visibility="Same real camera, native sprite sorting and alpha-aware conservative opaque label pass; full-screen top-down rows. No scene-wide actor/collider data in this envelope.",geometry="Object bounds are AABBs of actually surviving opaque sprite pixels, clipped to screen, not collider bounds; terrain labels describe visible appearance, not guaranteed collision.",unsupported="Alpha edges/translucency/unsupported shaders/masked sprites are unknown. No sprite means no visible object. Secret identity, enemy HP, exact own velocity/grounded and unrendered collider semantics are omitted. Unsupported non-sprite renderers not yet accepted.",resolution="Cell pixels="+cell+". Every per-cell semantic is retained as a bit-union and assigned a compact glyph by the Python adapter; unknown never implies empty.",debug="Full original global native state is separate debug_observation.json; it is not the default model payload."}};
    }
}
