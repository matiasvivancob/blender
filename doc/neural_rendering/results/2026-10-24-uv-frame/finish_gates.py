"""Finish UV-frame qualification serially; no edits to owner scenes or remote state."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

R = Path('D:/Github/Blender-RND')
S = R / 'blender'
W = Path('C:/Users/Matia/workspace/uv-frame-fix')
O = R / 'real/uv_frame_p4/final_gates'
T = S / 'doc/neural_rendering/tools/buildbox'
B = R / 'install/neural-p4/blender.exe'
MODEL = R / 'real/uv_frame_p4/cull_w32/model_proxy.nnm'
EXPECTED = {'cycles_motion_blur_cpu', 'cycles_motion_blur_cuda', 'cycles_motion_blur_optix', 'cycles_shadow_catcher_optix'}
env = dict(os.environ, PYTHONUNBUFFERED='1')
for key, leaf in [('BLENDER_USER_CONFIG','config'),('BLENDER_USER_SCRIPTS','scripts'),('BLENDER_USER_EXTENSIONS','extensions')]:
    path = W / leaf
    path.mkdir(exist_ok=True)
    env[key] = str(path)
state = {'status':'running','stages':[]}


def save():
    (O/'status.json').write_text(json.dumps(state,indent=2))


def run(name,cmd,cwd=R,extra=None,marker=None,allowed=(0,)):
    row={'name':name,'status':'running','command':list(map(str,cmd))}
    state['stages'].append(row)
    state['current']=name
    save()
    print('START',name,flush=True)
    start=time.perf_counter()
    log=O/(name+'.log')
    with log.open('w',encoding='utf-8') as f:
        result=subprocess.run(list(map(str,cmd)),cwd=cwd,env=dict(env,**(extra or {})),stdout=f,stderr=subprocess.STDOUT,
                              creationflags=subprocess.BELOW_NORMAL_PRIORITY_CLASS)
    text=log.read_text(encoding='utf-8',errors='replace')
    row.update(exit=result.returncode,seconds=time.perf_counter()-start)
    row['status']='passed' if result.returncode in allowed and (marker is None or marker in text) else 'failed'
    save()
    print('END',name,row['status'],flush=True)
    assert row['status']=='passed', f'{name}: {log}'
    return text


def blender(scene):
    return [B,'-b','--factory-startup','--disable-autoexec',scene,'--python-exit-code','1']


if __name__=='__main__':
    O.mkdir(parents=True,exist_ok=False)
    try:
        run('build_both',[sys.executable,R/'validation_build.py','sm_86;sm_89'])
        run('unit',['ctest','-R','^cycles$','--output-on-failure'],cwd=R/'build_windows_neural',marker='100% tests passed')
        run('depcheck',[R/'venv-neural/Scripts/python.exe',T/'iter_tools.py','depcheck',R/'build_windows_neural'],marker='depcheck PASS')
        for device,gpu in [('CPU',''),('CUDA','3090'),('CUDA','4070'),('OPTIX','3090'),('OPTIX','4070')]:
            name='furnace_'+device+gpu
            dest=O/name
            run(name,blender(R/'scenes/neural_material_probe.blend')+['--python',T/'neural_vs_principled.py','--',dest],
                extra={'NR_MODEL':str(MODEL),'NR_DEVICE':device,'NR_GPU':gpu,'NR_SPP':'256','NR_SEEDS':'4'},marker='FURNACE PASS')
            result=json.loads((dest/'furnace.json').read_text())
            assert result['gate_mean_within_3pct'] and result['gate_no_energy_gain'],name
        scene=O/'mis_neural.blend'
        run('make_mis_scene',blender(R/'scenes/mis_sphere.blend')+['--python',T/'make_neural_mis_scene.py','--',scene],
            extra={'NR_MODEL':str(MODEL)})
        for device,gpu in [('CPU',''),('OPTIX','4070')]:
            run('mis_'+device,blender(scene)+['--python',T/'mis_consistency.py','--','--device',device,
                *(['--gpu-name',gpu] if gpu else []),'--samples','1024','--out-dir',O/('mis_'+device)],marker='### MIS consistency: PASS')
        run('regression',['ctest','-R','^cycles_','--output-on-failure','--output-junit',O/'regression.xml'],
            cwd=R/'build_windows_neural',allowed=(0,8))
        cases=ET.parse(O/'regression.xml').getroot().findall('.//testcase')
        assert len(cases)==120 and not any(c.find('skipped') is not None for c in cases)
        failed={c.attrib['name'] for c in cases if c.find('failure') is not None or c.find('error') is not None}
        assert failed<=EXPECTED,sorted(failed)
        state['accepted_failures']=sorted(failed)
        state['status']='complete'
        save()
        print('UV FRAME FINAL GATES COMPLETE',flush=True)
    except BaseException as e:
        state['status']='failed'
        state['error']=repr(e)
        save()
        raise
