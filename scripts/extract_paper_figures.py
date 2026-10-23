"""Render exact figure regions from the supplied 40-page 2026-10-21 PDF.

Requires optional PyMuPDF. No plotting, interpolation of data, or AI redrawing.
Coordinates are PDF points, with page numbers counted from the PDF's first page.
"""
import argparse
import hashlib
import json
from pathlib import Path

# Figure number, manuscript label, physical PDF page, top, bottom, and title.
# The crop includes panel labels, color legends, and the manuscript caption.
REGIONS=[
    (1,'Figure 1:',4,347,510,'Conservative annotation'),
    (2,'Figure 2:',8,122,402,'Supervision mask generation'),
    (3,'Figure 3:',9,122,385,'Training and inference'),
    (4,'Figure 4:',11,122,366,'Local supervision changes'),
    (5,'Figure 5:',16,217,471,'Road recovery and boundary expansion'),
    (6,'Figure 6:',19,219,572,'Confidence and reliability diagnostics'),
    (7,'Figure 7:',20,122,396,'Fixed-quota negative selection'),
    (8,'Figure 8:',21,425,627,'Reconstruction benefit and training cost'),
    (9,'Figure 9:',22,235,437,'Other annotation and domain settings'),
    (10,'Figure 10:',24,325,536,'Latency and throughput'),
    (11,'Figure E.11:',31,122,346,'SegMine paired RGB and measured point clouds'),
    (12,'Figure F.12:',33,244,515,'Validation trajectories'),
    (13,'Figure F.13:',34,122,431,'SegMine segmentation examples'),
    (14,'Figure F.14:',34,446,668,'Left-edge recovery ROI'),
    (15,'Figure F.15:',35,165,444,'Quota and false negative supervision'),
    (16,'Figure F.16:',36,237,439,'Scene-wise gains'),
    (17,'Figure F.17:',37,122,436,'RELLIS-3D segmentation examples'),
]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--pdf',required=True)
    parser.add_argument('--output',default='docs/figures')
    parser.add_argument('--dpi',type=int,default=240)
    args=parser.parse_args()
    import fitz
    pdf=Path(args.pdf); output=Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    document=fitz.open(pdf)
    if len(document)!=40:
        raise ValueError('Crop map is only for the supplied 40-page manuscript')
    items=[]
    for number,label,page,top,bottom,title in REGIONS:
        clip=fitz.Rect(104,top,505,bottom)
        region_text=document[page-1].get_text(clip=clip)
        if label not in region_text:
            raise ValueError(f'{label} caption not found in crop')
        target=output/f'figure_{number:02d}.png'
        document[page-1].get_pixmap(matrix=fitz.Matrix(args.dpi/72,args.dpi/72),clip=clip,alpha=False).save(target)
        items.append(dict(figure=number,label=label[:-1],title=title,page=page,crop_points=list(clip),
                          file=target.name,sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
    manifest=dict(source_pdf=pdf.name,source_sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),
                  dpi=args.dpi,method='Exact PDF-region rendering, including original legends and captions',figures=items)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(f'Rendered {len(items)} figures from {pdf.name}')

if __name__=='__main__':
    main()
