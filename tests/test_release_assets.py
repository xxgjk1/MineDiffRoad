"""Release asset, configuration, and documentation checks."""
import csv
import hashlib
import json
from pathlib import Path
import re
import unittest
from urllib.parse import unquote
from minediffroad.engine import load_config

ROOT=Path(__file__).resolve().parents[1]


class ReleaseAssetsTests(unittest.TestCase):
    def test_assets_match_recorded_hashes(self):
        provenance=json.loads((ROOT/'docs/ASSET_PROVENANCE.json').read_text(encoding='utf-8'))
        self.assertEqual(len(provenance['figures']),17)
        self.assertEqual(len(provenance['results']),11)
        self.assertEqual([x['figure'] for x in provenance['figures']],list(range(1,18)))
        figure_manifest=json.loads((ROOT/'docs/figures/manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(provenance['source']['pdf_sha256'],figure_manifest['source_sha256'])
        for item in provenance['figures']+provenance['results']:
            p=ROOT/item['file']
            self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),item['sha256'],str(p))

    def test_fixed_quota_results(self):
        with (ROOT/'docs/results/table_2a_negative_placement.csv').open(encoding='utf-8',newline='') as f:
            rows=list(csv.DictReader(f))
        self.assertEqual([float(r['FNS ↓']) for r in rows],[2.13,2.16,2.03,1.72])
        self.assertTrue(all(float(r['Quota (%)'])==60 for r in rows))
        with (ROOT/'docs/results/table_2b_reliability_inputs.csv').open(encoding='utf-8',newline='') as f:
            self.assertEqual(next(csv.reader(f)),['Ranking signal','IoU ↑'])

    def test_current_docs_local_links(self):
        paths=[ROOT/'README.md',ROOT/'README.zh-CN.md']+list((ROOT/'docs').glob('*.md'))+[ROOT/'docs/results/README.md']
        for path in paths:
            for target in re.findall(r'\]\(([^)]+)\)',path.read_text(encoding='utf-8')):
                if '://' in target or target.startswith('#'):
                    continue
                relative=unquote(target.split('#')[0])
                self.assertTrue((path.parent/relative).exists(),f'{path.name}: {target}')

    def test_controls_preserve_paper_settings(self):
        for name in ('conf','concat'):
            c=load_config(ROOT/f'configs/{name}.yaml')
            self.assertEqual(c['selection']['negative_rule'],'confidence')
            self.assertEqual(c['selection']['outer'],25)
            self.assertEqual(c['selection']['unknown_area_cap'],.3)
            self.assertEqual(c['image_size'],1024)
            self.assertEqual(c['depth_tolerance_m'],0)
            self.assertEqual(c['batch_size'],2)
            self.assertEqual(c['learning_rate'],1e-3)
            self.assertEqual(c['weight_decay'],1e-4)
            self.assertEqual(c['learning_rate_schedule'],'constant')
        self.assertEqual(load_config(ROOT/'configs/concat.yaml')['feature_fusion'],'concat')
        matched=load_config(ROOT/'configs/conf_matched.yaml')
        self.assertEqual(matched['selection']['negative_rule'],'confidence_matched')
        self.assertEqual(matched['selection']['tau_n'],load_config(ROOT/'configs/paper.yaml')['selection']['tau_n'])
        self.assertEqual(matched['batch_size'],2)

    def test_manifest_identifies_current_source_and_optimizer(self):
        source=json.loads((ROOT/'docs/SOURCE_SPEC.json').read_text(encoding='utf-8'))
        self.assertEqual(source['sources'][-2]['sha256'],
                         'd38ed19d3da2d18f08ab4d84d7d0bb03156bfaaee1fa1c3942a977b8f05a093b')
        self.assertEqual(source['sources'][-1]['sha256'],
                         '04d8770239f26c1268c9d1f6902718ad1c086caf8e2876d5579ef2771004d786')
        config=load_config(ROOT/'configs/paper.yaml')
        example=load_config(ROOT/'configs/example.yaml')
        self.assertEqual(source['default_optimizer']['name'],'AdamW')
        for key in ('learning_rate','weight_decay','learning_rate_schedule','batch_size'):
            self.assertEqual(source['default_optimizer'][key],config[key])
            self.assertEqual(example[key],config[key])
