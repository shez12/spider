"""Read scalar history through the W&B GraphQL schema used by its public SDK.

Uses existing credentials only for api.wandb.ai; never prints credentials.
Avoids a stalled SDK service process when verifying an upload.
"""
import json,netrc,os
import requests


def history_rows(url,keys,samples=10000):
    path=url.removeprefix('https://wandb.ai/').replace('/runs/','/').split('/')
    assert len(path)==3
    entity,project,run_id=path
    key=os.environ.get('WANDB_API_KEY')
    if not key:
        auth=netrc.netrc().authenticators('api.wandb.ai')
        assert auth,'No configured W&B credentials.'
        key=auth[2]
    session=requests.Session();session.auth=('api',key)
    query='''query RunSampledHistory($project:String!,$entity:String!,$name:String!,$specs:[JSONString!]!){
      project(name:$project,entityName:$entity){run(name:$name){sampledHistory(specs:$specs)}}}'''
    response=session.post('https://api.wandb.ai/graphql',json=dict(query=query,variables=dict(
        project=project,entity=entity,name=run_id,specs=[json.dumps(dict(keys=keys,samples=samples))])),timeout=60)
    response.raise_for_status();payload=response.json()
    if payload.get('errors'):raise RuntimeError(str(payload['errors']))
    run=payload['data']['project']['run']
    if run is None:return []
    rows=run['sampledHistory'][0]
    if isinstance(rows,str):rows=json.loads(rows)
    return rows
