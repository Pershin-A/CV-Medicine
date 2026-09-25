"""Browser regression for v5.3: freehand pens + independent image thresholds.

Run from project root: python tests/browser_smoke.py
Requires: pip install playwright pillow && python -m playwright install chromium
Uses a synthetic image, never a patient DICOM.
"""
import base64
import io
import shutil
from pathlib import Path

from PIL import Image, ImageDraw
from playwright.sync_api import sync_playwright

image = Image.new("L", (300, 500), 30)
ImageDraw.Draw(image).rectangle((80, 70, 215, 430), fill=130)
buffer = io.BytesIO()
image.save(buffer, format="PNG")
geometry = {
    "schema_version": 1,
    "coordinate_system": "original_dicom_pixels_top_left",
    "image_width": 600,
    "image_height": 1000,
    "spine": {"disc_lines": [], "iliac_crests": {"image_left": None, "image_right": None},
              "foreign_objects": []},
    "hip": {"landmarks": {"greater_trochanter": None, "femoral_neck": None,
                          "ischial_bone": None}, "lesser_trochanter": None, "roi_box": None},
    "complete": {"spine": False, "hip": False},
}
args = dict(source_key="synthetic", region="SPINE", native_width=600,
            native_height=1000, preview_width=300, preview_height=500,
            image_png_base64=base64.b64encode(buffer.getvalue()).decode(), geometry=geometry)
html = (Path(__file__).resolve().parents[1] / "dxa_canvas" / "index.html").read_text(encoding="utf-8")

with sync_playwright() as playwright:
    browser = playwright.chromium.launch(
        headless=True, executable_path=shutil.which("chromium") or None,
        args=["--no-sandbox"],
    )
    page = browser.new_page(viewport={"width": 820, "height": 1450})
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.set_content(html)
    page.evaluate("""() => {
       window.testValues=[];
       window.addEventListener('message',e=>{
         if(e.data.type==='streamlit:setComponentValue')window.testValues.push(e.data.value);
       });
    }""")
    page.evaluate('arg=>window.postMessage({type:"streamlit:render",args:arg},"*")', args)
    page.wait_for_function('document.querySelector("#canvas").style.width.length>0')

    def coords():
        return page.evaluate("""() => {
            const r=document.querySelector('#viewport').getBoundingClientRect();
            const canvas=document.querySelector('#canvas').getBoundingClientRect();
            const tools=document.querySelector('#toolsSecondary').getBoundingClientRect();
            const items=document.querySelector('#items').getBoundingClientRect();
            return {top:r.top,viewport_height:r.height,display_width:canvas.width,
                    display_height:canvas.height,second_row_top:tools.top,
                    items_top:items.top,items_height:items.height,
                    frame_height:document.body.scrollHeight};
        }""")

    def click_native(x, y):
        rect = page.locator("#canvas").bounding_box()
        page.mouse.click(rect["x"] + rect["width"] * x / 600,
                         rect["y"] + rect["height"] * y / 1000)

    def stroke_native(points):
        rect = page.locator('#canvas').bounding_box()
        def pos(p):
            return (rect['x']+rect['width']*p[0]/600,
                    rect['y']+rect['height']*p[1]/1000)
        page.mouse.move(*pos(points[0]))
        page.mouse.down()
        for p in points[1:]:
            page.mouse.move(*pos(p),steps=3)
        page.mouse.up()

    def last():
        return page.evaluate("window.testValues.at(-1)")

    def chip_count():
        return page.locator("#items .item").count()

    # Stable initial page: fixed-size object row is present while empty.
    initial = coords()
    assert initial["viewport_height"] == 820, initial
    assert initial["display_height"] > 780, initial
    assert chip_count() == 0
    assert initial["items_height"] == 39
    assert page.locator("#foreignLabel").evaluate("e=>getComputedStyle(e).visibility") == "hidden"

    # A line takes exactly TWO single clicks (one start, one end).
    click_native(170, 210)
    assert len(page.evaluate("window.testValues")) == 0
    click_native(420, 215)
    page.wait_for_function('window.testValues.length===1')
    assert len(last()["spine"]["disc_lines"]) == 1
    assert chip_count() == 1
    assert coords()["top"] == initial["top"], "Image shifted after adding first object"
    assert coords()["items_top"] == initial["items_top"], "Items row shifted"
    # Emulate Streamlit echo after each commit; the editor must not re-create
    # tools or require a second click once parent rerenders the same image.
    args["geometry"] = last()
    page.evaluate('arg=>window.postMessage({type:"streamlit:render",args:arg},"*")', args)
    page.wait_for_function('document.querySelector("[data-tool=disc]").getAttribute("aria-pressed")==="true"')
    assert chip_count() == 1
    assert coords()["top"] == initial["top"]

    # A single pointer press sets a point and adds it to the persistent object row.
    page.locator('button[data-tool="iliac_left"]').click()
    click_native(65, 850)
    page.wait_for_function('window.testValues.length===2')
    assert last()["spine"]["iliac_crests"]["image_left"] == [65, 850]
    assert chip_count() == 2
    assert coords()["top"] == initial["top"]
    page.locator('button[data-tool="iliac_right"]').click()
    click_native(530, 850)
    assert chip_count() == 3

    # Foreign-kind slot lives after Foreign and Erase tools in row #2,
    # and its visibility changes without moving viewport/image.
    page.locator('button[data-tool="foreign"]').click()
    assert page.locator("#foreignLabel").evaluate("e=>getComputedStyle(e).visibility") == "visible"
    positions = page.evaluate("""() => ['foreign','erase'].map(t=>
       document.querySelector('[data-tool="'+t+'"]').getBoundingClientRect().left)
       .concat(document.querySelector('#foreignLabel').getBoundingClientRect().left)""")
    assert positions == sorted(positions), positions
    assert coords()["top"] == initial["top"]
    slot=page.locator("#foreignLabel").bounding_box()
    button=page.locator('button[data-tool="foreign"]').bounding_box()
    assert abs((slot["y"]+slot["height"]/2)-(button["y"]+button["height"]/2))<2

    # Rectangle takes TWO single clicks, without press-and-drag.
    count_before = len(page.evaluate("window.testValues"))
    click_native(100, 60)
    assert len(page.evaluate("window.testValues")) == count_before
    click_native(210, 150)
    page.wait_for_function('window.testValues.at(-1).spine.foreign_objects.length===1')
    assert last()["spine"]["foreign_objects"][0]["bbox"] == [100, 60, 210, 150]
    assert chip_count() == 4
    assert coords()["top"] == initial["top"]

    # Erase uses one click; deleting a chip never collapses the items row.
    page.locator('button[data-tool="erase"]').click()
    assert page.locator("#foreignLabel").evaluate("e=>getComputedStyle(e).visibility") == "hidden"
    click_native(160, 100)
    page.wait_for_function('window.testValues.at(-1).spine.foreign_objects.length===0')
    assert last()["spine"]["foreign_objects"] == []
    assert chip_count() == 3
    assert coords()["top"] == initial["top"]
    assert coords()["items_height"] == initial["items_height"]

    # Switch to hip tools and verify all landmarks, both fine-pen layers and ROI.
    args["region"] = "LEG"
    args["geometry"] = last()
    page.evaluate('arg=>window.postMessage({type:"streamlit:render",args:arg},"*")', args)
    page.wait_for_function('document.querySelector("[data-tool=greater]")!==null')
    hip_initial = coords()
    for key, xy in [("greater", (70, 90)), ("neck", (150, 155)), ("ischial", (260, 190))]:
        page.locator(f'button[data-tool="{key}"]').click()
        click_native(*xy)
        label={"greater":"greater_trochanter","neck":"femoral_neck","ischial":"ischial_bone"}[key]
        page.wait_for_function('k=>window.testValues.at(-1).hip.landmarks[k]!==null',arg=label)
    assert all(last()["hip"]["landmarks"].values())
    assert chip_count() == 3
    page.locator('button[data-tool="trace_trochanter"]').click()
    stroke_native([(100,230),(115,234),(140,245),(146,270),(110,265)])
    page.wait_for_function('window.testValues.at(-1).hip.lesser_trochanter_traces.trochanter.length===1')
    t=last()['hip']['lesser_trochanter_traces']['trochanter'][0]
    assert len(t['points'])>5 and t['points'][0]==[100,230]
    assert chip_count() == 4
    page.locator('button[data-tool="trace_bone"]').click()
    stroke_native([(160,225),(168,243),(174,267),(166,280)])
    page.wait_for_function('window.testValues.at(-1).hip.lesser_trochanter_traces.adjacent_bone.length===1')
    assert chip_count()==5
    assert last()['hip']['lesser_trochanter'] is None

    # Repeated brush strokes and a single-click deletion, even with stale echoes.
    old=last()
    page.locator('button[data-tool="trace_trochanter"]').click()
    stroke_native([(111,240),(121,249),(131,260)])
    page.wait_for_function('window.testValues.at(-1).hip.lesser_trochanter_traces.trochanter.length===2')
    args['geometry']=old
    page.evaluate('arg=>window.postMessage({type:"streamlit:render",args:arg},"*")',args)
    assert chip_count()==6
    page.locator('#items .item').filter(has_text='Малый вертел · штрих 2').locator('button').click()
    page.wait_for_function('window.testValues.at(-1).hip.lesser_trochanter_traces.trochanter.length===1')
    assert chip_count()==5
    page.locator('button[data-tool="roi"]').click()
    click_native(40, 80)
    click_native(220, 440)
    page.wait_for_function('window.testValues.at(-1).hip.roi_box!==null')
    assert last()["hip"]["roi_box"] == [40, 80, 220, 440]
    assert chip_count() == 6
    assert coords()["top"] == hip_initial["top"]
    assert coords()["items_top"] == hip_initial["items_top"]
    assert coords()["items_height"] == 39

    # A distinct image-view mode: only the bright threshold mask is colored,
    # with a narrow edge, and the per-image threshold travels in the JSON.
    page.locator('#viewMode').select_option('threshold')
    assert last()['image_view']=={'mode':'threshold','threshold_8bit':128}
    page.locator('#threshold').evaluate("e=>{e.value='90';e.dispatchEvent(new Event('input',{bubbles:true}));e.dispatchEvent(new Event('change',{bubbles:true}));}")
    page.wait_for_function('window.testValues.at(-1).image_view.threshold_8bit===90')
    assert last()['image_view']=={'mode':'threshold','threshold_8bit':90}
    def pixel(x,y):
        return page.evaluate('([x,y])=>Array.from(document.querySelector("#canvas").getContext("2d").getImageData(x,y,1,1).data)',[x,y])
    inside=pixel(150,300)
    outside=pixel(10,300)
    assert inside[1]>130 and outside[:3]==[30,30,30],(inside,outside)
    assert pixel(80,300)[0]>200, 'Boundary must be visibly outlined'
    page.screenshot(path=str(Path(__file__).resolve().parents[1] / 'tests' / 'hip_freehand_threshold_v53.png'),full_page=True)
    page.locator('#viewMode').select_option('original')
    assert last()['image_view']=={'mode':'original','threshold_8bit':90}
    assert pixel(150,300)[:3]==[130,130,130]
    page.locator('#viewMode').select_option('threshold')
    assert last()['image_view']['threshold_8bit']==90

    # Zoom magnifies actual displayed image; native stored coordinates remain unchanged.
    before_zoom = coords()["display_width"]
    page.locator("#zoom").select_option("1.5")
    assert coords()["display_width"] > before_zoom * 1.45
    assert last()["hip"]["roi_box"] == [40, 80, 220, 440]

    # Regression for stale Streamlit renders: after two or more edits a delayed
    # parent echo used to resurrect removed objects / erase newly added ones.
    # Replay them in both orders while checking the actual iframe canvas state.
    args['region']='SPINE'
    page.evaluate('arg=>window.postMessage({type:"streamlit:render",args:arg},"*")', args)
    page.wait_for_function('document.querySelector("[data-tool=disc]")!==null')
    page.locator('button[data-tool="disc"]').click()
    before_add=last()
    preexisting=chip_count()
    for i in range(3):
        click_native(120, 250+110*i)
        # Delayed render in the middle of a two-click action must not remove
        # its pending starting point or revert any previously added line.
        args['geometry']=before_add
        page.evaluate('arg=>window.postMessage({type:"streamlit:render",args:arg},"*")', args)
        click_native(420, 252+110*i)
        page.wait_for_function('n=>window.testValues.at(-1).spine.disc_lines.length===n', arg=2+i)
        assert chip_count()==preexisting+i+1
        args['geometry']=before_add
        page.evaluate('arg=>window.postMessage({type:"streamlit:render",args:arg},"*")', args)
        page.wait_for_timeout(70)
        assert chip_count()==preexisting+i+1, 'Stale parent snapshot rolled back a new line'
        assert len(last()['spine']['disc_lines'])==2+i
        before_add=last()

    previous_revisions=[v['_client_revision'] for v in page.evaluate('window.testValues')]
    assert previous_revisions==sorted(set(previous_revisions)), 'Event revisions must be strictly increasing'
    assert chip_count()>=6

    # Delete multiple lines with a SINGLE click on the × in the object row.
    # Echoing the pre-delete snapshot must not bring any deleted line back.
    for expected in (3,2,1,0):
        before_delete=last()
        page.locator('#items .item').first.locator('button').click()
        page.wait_for_function('n=>window.testValues.at(-1).spine.disc_lines.length===n', arg=expected)
        args['geometry']=before_delete
        page.evaluate('arg=>window.postMessage({type:"streamlit:render",args:arg},"*")', args)
        page.wait_for_timeout(70)
        assert len(last()['spine']['disc_lines'])==expected
        assert chip_count()==expected+2, 'Deleted object reappeared after stale render'

    # Two late echoes in reverse order cannot resurrect the removed set.
    args['geometry']=before_add
    page.evaluate('arg=>window.postMessage({type:"streamlit:render",args:arg},"*")', args)
    page.wait_for_timeout(70)
    assert len(last()['spine']['disc_lines'])==0
    assert chip_count()==2

    # Replacing an existing landmark needs one click, with no flicker / rollback
    # when Streamlit replays the earlier snapshot.
    page.locator('#zoom').select_option('1')
    page.locator('button[data-tool="iliac_left"]').click()
    old_points=last()
    click_native(85, 780)
    page.wait_for_function('window.testValues.at(-1).spine.iliac_crests.image_left[0]===85')
    args['geometry']=old_points
    page.evaluate('arg=>window.postMessage({type:"streamlit:render",args:arg},"*")', args)
    page.wait_for_timeout(70)
    assert last()['spine']['iliac_crests']['image_left']==[85,780]
    assert chip_count()==2
    older_points=last()
    click_native(95, 785)
    args['geometry']=older_points
    page.evaluate('arg=>window.postMessage({type:"streamlit:render",args:arg},"*")', args)
    page.wait_for_timeout(70)
    assert last()['spine']['iliac_crests']['image_left']==[95,785]
    assert chip_count()==2

    # Delete remaining point chips exactly once each, even with delayed echoes.
    for count in (1,0):
        point_snapshot=last()
        page.locator('#items .item').first.locator('button').click()
        assert chip_count()==count
        args['geometry']=point_snapshot
        page.evaluate('arg=>window.postMessage({type:"streamlit:render",args:arg},"*")', args)
        page.wait_for_timeout(70)
        assert chip_count()==count
    assert last()['spine']['iliac_crests']=={'image_left':None,'image_right':None}

    # Different image = different saved brightness threshold, no shared slider.
    first_saved=last()
    second=geometry.copy()
    second['image_view']={'mode':'threshold','threshold_8bit':211}
    args.update(source_key='synthetic-other',geometry=second)
    page.evaluate('arg=>window.postMessage({type:"streamlit:render",args:arg},"*")',args)
    page.wait_for_function('document.querySelector("#threshold").value==="211"')
    assert page.locator('#viewMode').input_value()=='threshold'
    args.update(source_key='synthetic',geometry=first_saved)
    page.evaluate('arg=>window.postMessage({type:"streamlit:render",args:arg},"*")',args)
    page.wait_for_function('document.querySelector("#threshold").value==="90"')
    assert page.locator('#viewMode').input_value()=='threshold'

    page.screenshot(path=str(Path(__file__).resolve().parents[1] / "tests" / "synthetic_browser_v53.png"), full_page=True)
    assert not errors, errors
    print("BROWSER REGRESSION PASSED: thin mouse-drag traces (both layers); brightness threshold, image-specific restoration, single clicks, delayed echoes, one-click deletes, zoom")
    browser.close()
