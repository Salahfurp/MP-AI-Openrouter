"""Offline smoke tests for the deterministic core. Run: python SELF_TEST.py"""
from pathlib import Path
from public_facilities_engine_v2 import review_public_facilities
from public_facilities_comparison import SubmittedFacility, best_official_service_match, compare_facilities, parse_excel_bytes

root=Path(__file__).parent
required=review_public_facilities(850000,25000)
assert len(required.get('required',[])) >= 1
assert best_official_service_match('Secondary Education School')[0] == 'مدرسة ثانوي'
assert best_official_service_match('Friday Mosque')[0] == 'مسجد جمعة'
assert best_official_service_match('Neighbourhood Park')[0] == 'حديقة حي'
submitted=[SubmittedFacility(service='Secondary Education School',count=1),SubmittedFacility(service='Friday Mosque',count=2),SubmittedFacility(service='Neighbourhood Park',count=1)]
report=compare_facilities(required,submitted)
assert report.get('summary',{}).get('required_services',0) >= 1
# Parse a generated consultant-style English workbook entirely in memory.
import pandas as pd, io
buf=io.BytesIO()
pd.DataFrame([['Provided Public Facilities','',''],['Service','Count','Land Area'],['Secondary Education School',1,6500],['Friday Mosque',2,5100],['Neighbourhood Park',1,8800]]).to_excel(buf,index=False,header=False)
parsed=parse_excel_bytes(buf.getvalue())
assert any(best_official_service_match(r.service)[0]=='مدرسة ثانوي' for r in parsed)
print('SELF_TEST_OK')
