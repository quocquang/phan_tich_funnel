# -*- coding: utf-8 -*-
"""BÁO CÁO TUYỂN DỤNG THEO THÁNG / TUẦN – một file duy nhất.

CÁCH DÙNG (chỉ 3 bước):
  1. Chạy app:   streamlit run app_tuyen_dung.py      (hoặc đưa file này lên Streamlit Cloud)
  2. Kéo thả 3–4 file raw vào ô "Kéo thả các file raw" ở thanh bên trái:
       DATA tuyển dụng | Kế toán Admin Update | Master Data | Report Facebook Ads (không bắt buộc)
     Tên file và thứ tự tùy ý – app tự nhận ra file nào là file nào theo nội dung.
  3. Chọn năm + các tháng cần xem. Tải Excel ở nút trên thanh bên.
(Không cần thư mục. Chỉ khi muốn đọc file có sẵn trên máy thì chọn "Thư mục trên máy" hoặc đặt biến môi trường TD_DATA_DIR.)

Cần cài: pip install streamlit pandas numpy openpyxl xlsxwriter

Định nghĩa (đúng công thức của file DATA / báo cáo tháng 10):
  CV = ngày liên lạc trong tháng | Tham gia PV = ngày PV trong tháng & KQ ∈ {Đạt, Đợi kết quả, Không đạt} | Pass = ngày PV trong tháng & Đạt
  Nhận việc = ngày nhận việc trong tháng & "Đã nhận việc" | Nghỉ việc = hợp CHECK LIST + NGHỈ VIỆC (Kế toán) + Master theo ngày nghỉ; nghỉ thử việc = làm < 30 ngày.
Công tắc "Khớp báo cáo đã duyệt" áp dụng khối điều chỉnh ADJ (bên dưới) cho tháng có báo cáo; tắt = số thuần từ raw."""
import io, os, re, glob, datetime, unicodedata, difflib, warnings
import numpy as np, pandas as pd
import streamlit as st
try:
    import openpyxl, xlsxwriter   # openpyxl: đọc file raw; xlsxwriter: xuất Excel
except ImportError as _ex:
    st.error(f'Thiếu thư viện: {_ex.name}. Thêm vào requirements.txt (streamlit, pandas, numpy, openpyxl, xlsxwriter) rồi chạy lại.'); st.stop()
warnings.filterwarnings('ignore')

# =====================================================================================================
# 1) ĐIỀU CHỈNH ĐỂ KHỚP BÁO CÁO ĐÃ DUYỆT (mỗi tháng một khối; tháng không có khối = dùng số raw)
# =====================================================================================================
LY_RAW = 'DATA raw thấp hơn báo cáo, chưa xác định được dòng (nghi do dòng Đợi kết quả / thiếu ngày PV được cập nhật sau)'
ADJ = {
 (2026, 10): dict(
  ten='Báo cáo tuyển dụng tháng 10/2026 (đã duyệt)',
  nhan_viec_bo_sung=[
    dict(TEN='NGUYỄN THỊ YẾN LINH', KHOI='Chi Nhánh', CN='Vũng Tàu', VITRI=None, NGUON='Giới thiệu nội bộ', LH='Sương', NNV='2026-10-31', ly_do='DATA để trống kết quả nhận việc (báo cáo tính đã nhận việc)'),
    dict(TEN='PHẠM THỊ PHƯƠNG NHUNG', KHOI='Chi Nhánh', CN='Thủ Đức', VITRI=None, NGUON='Zalo', LH='Sương', NNV='2026-10-23', ly_do='DATA chỉ có bản ghi ngày NV 12/09/2026 (báo cáo tính nhận việc 23/10)')],
  loai_tru_nghi=[
    ('NGUYEN THANH THAO', '2026-10-26', 'Có trong Kế toán/DATA là đã nghỉ 27/10 nhưng báo cáo không liệt kê – CẦN XÁC MINH'),
    ('TRINH NGUYEN HOANG YEN', '2026-10-14', 'Nghỉ 20/10 theo Kế toán, báo cáo không liệt kê – CẦN XÁC MINH'),
    ('TRUONG MY PHUONG', '2026-10-14', 'Nghỉ 20/10 theo Kế toán, báo cáo không liệt kê – CẦN XÁC MINH'),
    ('TRAN THI THANH THAO', '2026-10-12', 'Bình Dương, nghỉ ngang 13/10, báo cáo không liệt kê – CẦN XÁC MINH')],
  delta_nguon={('Zalo','CV'):1, ('(không ghi nguồn)','CV'):-1, ('Zalo','PV'):1, ('Zalo','PASS'):3, ('Zalo','NV'):1, ('Giới thiệu nội bộ','NV'):1, ('Giới thiệu nội bộ','KHONG_NV'):-1},
  delta_hr={('Thảo','CV'):1, ('(không ghi người LH)','CV'):-1, ('Thảo','PV'):1, ('Sương','PASS'):2, ('Thảo','PASS'):1, ('Sương','NV'):1, ('Thảo','NV'):1},
  delta_unit={('Bình Dương','PASS'):2, ('Vũng Tàu','PASS'):1, ('Vũng Tàu','PV'):1, ('Vũng Tàu','NV'):1, ('Thủ Đức','NV'):1, ('Vũng Tàu','KHONG_NV'):-1},
  delta_dept={('TÂN BÌNH','KDAT'):1, ('BÌNH DƯƠNG','DAT'):2, ('VŨNG TÀU','CV'):1, ('VŨNG TÀU','PV'):1, ('VŨNG TÀU','DAT'):1, ('VŨNG TÀU','NV'):1, ('VŨNG TÀU','KNV'):-1, ('THỦ ĐỨC','NV'):1},
  ly_do_delta={'nguon':'Zalo +1 CV (dòng LƯU THỊ THANH THẢO thiếu nguồn/người LH), +1 PV, +3 Pass (2 dòng Bình Dương "Đợi kết quả" → Đạt và TRẦN HOÀI AN chưa có ngày PV/kết quả); chưa xác nhận.',
               'hr': LY_RAW, 'unit': LY_RAW, 'dept': LY_RAW + '. Riêng Tân Bình "Không đạt PV": báo cáo ghi 2 do công thức BC-2026!K26 trỏ nhầm cột, dữ liệu là 1.'},
  dau_ky={'QUẬN 1':68,'PHÚ NHUẬN':66,'TÂN BÌNH':35,'GÒ VẤP':68,'QUẬN 10':53,'TÂN PHÚ':48,'BÌNH TÂN':44,'VŨNG TÀU':19,'BÌNH DƯƠNG':39,'THỦ ĐỨC':19},
  sl_yc={'QUẬN 1':10,'PHÚ NHUẬN':5,'TÂN BÌNH':11,'GÒ VẤP':13,'QUẬN 10':10,'TÂN PHÚ':7,'BÌNH TÂN':10,'VŨNG TÀU':2,'BÌNH DƯƠNG':4,'THỦ ĐỨC':-6},
  chi_phi_tay={'CareerViet':7056000, 'TopCV':3370000},
  ads_bo_sung={'2026-10-30':(3134353,12), '2026-10-31':(2520176,9)},
  ly_do_nhom={'Gia đình, cá nhân, về quê':18,'Công việc không phù hợp':13,'Sức khỏe không đảm bảo':9,'Môi trường làm việc không phù hợp':7,'Thời gian làm việc, áp lực KPI':6,'Thay đổi định hướng, tìm cơ hội mới':5,'Khoảng cách, đi lại, điều chuyển chi nhánh':6,'Học tập, thi cử':1,'Công ty cho nghỉ':1,'Không tái ký hợp đồng':1,'Hết thực tập':1,'Không liên hệ được':3},
  baseline_raw=dict(funnel=dict(CV=643, PV=200, PASS=129, KHONG_DAT=56, DOI_KQ=15, KHONG_DEN_PV=33, NV=98, KHONG_NV=35), nghi_raw=74),   # số thuần từ raw lúc lập điều chỉnh: nếu raw đổi thì cảnh báo
  dat_thu_viec_bo_sung={'HM3188':'Master: đã nghỉ 07/10 sau khi hết thử việc 02/10 – báo cáo vẫn liệt kê'},
  cn_ghi_de={'LE HOANG MY':'Tân Bình','HO LE MY HOA':'Tân Bình'},
  ten_theo_bao_cao={'VO NGUYEN PHONG':'VŨ NGUYÊN PHONG','TRAN THI QUYNH NHI':'TRẦN QUỲNH NHI','NGUYEN THANH':'NGUYỄN VĂN THÀNH','BUI VO THUY VY':'BÙI VÕ THẢO VY','NGUYEN THI YEN NHI':'NGUYỄN THỊ YẾN NHƯ','TRUONG HOAI BAO NGOC':'TRƯƠNG HOÀNG BẢO NGỌC','NGO THI HANH THI':'NGÔ HỒ HẠNH THI'},
 ),
}
def get_adj(year, month): return ADJ.get((int(year), int(month)))


# =====================================================================================================
# 2) LÕI TÍNH TOÁN
# =====================================================================================================
PVOK = ['Đạt', 'Đợi kết quả', 'Không đạt']
DATA_COLS = ['STT','KHOI','CN','VITRI','TEN','MAIL','SDT','LINKCV','NGUON','DANHGIA','SOVAN','NGAYPV','GIOPV','NGUOIPV','LH','NGAYLL','DIACHI','KQPV','DGDG','LYDOFAIL','NGAYNV','KQNV','LYDONV','LUONG','TGTV','MANV','NGAYHH','TTHH','RUBY','HHTN','TTLV','NGAYNGHI','PLLD','LYDOCT']
BUCKETS = ['0 ngày','1–3 ngày','4–10 ngày','11–30 ngày','31–60 ngày','Trên 60 ngày']

# ------------------------------------------------------------------ tiện ích
def norm(s):
    if s is None or (isinstance(s, float) and pd.isna(s)): return ''
    s = str(s).strip().upper().replace('Đ', 'D'); s = unicodedata.normalize('NFD', s)
    s = ''.join(c for c in s if unicodedata.category(c) != 'Mn')
    return re.sub(r'\s+', ' ', re.sub(r'[^A-Z0-9 ]', ' ', s)).strip()
def pdate(x):
    """Ngày thật giữ nguyên; ô nhập dạng chữ đọc theo dd/mm/yyyy (không để pandas đoán tháng/ngày)."""
    if x is None or (isinstance(x, float) and pd.isna(x)): return pd.NaT
    if isinstance(x, str):
        m = re.search(r'(\d{1,2})\s*[/.-]\s*(\d{1,2})\s*[/.-]\s*(\d{4})', x)
        if not m: return pd.NaT
        try: return pd.Timestamp(int(m.group(3)), int(m.group(2)), int(m.group(1)))
        except Exception: return pd.NaT
    return pd.to_datetime(x, errors='coerce')
def txt(s):  # chuỗi sạch, ô trống -> None
    return s.where(s.notna(), None).map(lambda x: str(x).strip() if x is not None and str(x).strip() not in ('', 'nan') else None)
def pct(a, b): return (a / b * 100.0) if b else np.nan
def month_range(y, m):
    s = pd.Timestamp(int(y), int(m), 1); return s, s + pd.offsets.MonthEnd(0)
def _in(series, s, e): return (series >= s) & (series <= e)
def bucket(n):
    return '0 ngày' if n <= 0 else '1–3 ngày' if n <= 3 else '4–10 ngày' if n <= 10 else '11–30 ngày' if n <= 30 else '31–60 ngày' if n <= 60 else 'Trên 60 ngày'
def _b(x): return io.BytesIO(x) if isinstance(x, (bytes, bytearray)) else x
KIND_LABEL = {'data': 'TUYỂN DỤNG – DATA', 'kt': 'Kế toán Admin Update', 'master': 'Master Data', 'ads': 'Report Facebook Ads (tùy chọn)'}
def classify_file(src):
    """Nhận diện loại file raw theo NỘI DUNG (tên sheet / tiêu đề cột), không phụ thuộc tên file. Trả về (loại, ghi chú) – loại ∈ data/kt/master/ads hoặc None."""
    try: wb = openpyxl.load_workbook(_b(src), read_only=True, data_only=True)
    except Exception: return None, 'không mở được (không phải file Excel .xlsx hoặc file bị hỏng)'
    try:
        names = list(wb.sheetnames)
        if 'DATA TUYỂN DỤNG' in names: return 'data', ''
        if all(n in names for n in ('CHECK LIST NGHỈ VIỆC', 'NGHỈ VIỆC', 'NHẬN VIỆC')): return 'kt', ''
        if '1. TỔNG DATA - NHÂN SỰ' in names: return 'master', ''
        hdr = next(wb[names[0]].iter_rows(min_row=1, max_row=1, values_only=True), ())
        h = [norm(str(c)) for c in hdr if c is not None]
        if any(x.startswith('NGAY') for x in h) and any(('CHI TIEU' in x or 'SPENT' in x or 'AMOUNT' in x) for x in h): return 'ads', ''
        return None, 'không nhận ra loại file (sheet: ' + ', '.join(n.strip() for n in names[:4]) + ')'
    except Exception as ex: return None, f'không đọc được cấu trúc ({type(ex).__name__})'
    finally:
        try: wb.close()
        except Exception: pass
def find_files(folder):
    """Tìm file raw trong thư mục theo NỘI DUNG (không theo tên); nếu có nhiều file cùng loại lấy file mới nhất."""
    best = {}
    for f in glob.glob(os.path.join(folder, '*.xlsx')):
        if os.path.basename(f).startswith('~$'): continue
        kind, _ = classify_file(f)
        if kind and (kind not in best or os.path.getmtime(f) > os.path.getmtime(best[kind])): best[kind] = f
    return best

# ------------------------------------------------------------------ đọc raw
def kt_block(src, sheet, ncols):
    """Đọc sheet theo tuần của Kế toán; mỗi dòng trả thêm số dòng trong file gốc ở cuối."""
    ws = openpyxl.load_workbook(_b(src), data_only=True)[sheet]; out = []
    for i, r in enumerate(ws.iter_rows(values_only=True), start=1):
        if isinstance(r[0], str) and re.match(r'^\s*TU[ẤẦ]N\s*\d+', r[0].upper()): continue
        if r[1] == 'TÌNH TRẠNG' or r[0] == 'Chi nhánh' or all(c is None for c in r[:ncols]): continue
        out.append(tuple(r[:ncols]) + (i,))
    return out
def _open_xl(src, label, sheets):
    """Mở file Excel và báo lỗi tiếng Việt rõ ràng nếu thiếu sheet (thường do chọn nhầm file)."""
    try: xl = pd.ExcelFile(_b(src))
    except Exception: raise ValueError(f'Không mở được file {label} (có thể không phải file Excel .xlsx hoặc file bị hỏng).')
    miss = [n for n in sheets if n not in xl.sheet_names]
    if miss: raise ValueError(f"File {label} không có sheet {', '.join(repr(m) for m in miss)}. Sheet hiện có: {', '.join(xl.sheet_names[:12])}. Có phải bạn chọn nhầm file?")
    return xl
def _check_headers(cols, expect, label):
    bad = []
    for i, tok in expect.items():
        have = norm(str(cols[i]).replace('\n', ' ')) if i < len(cols) else '(thiếu cột)'
        if tok not in have: bad.append(f"cột {i + 1} là '{str(cols[i]).strip() if i < len(cols) else ''}' (cần chứa '{tok}')")
    if bad: raise ValueError(f'Cấu trúc cột của {label} khác mẫu: ' + '; '.join(bad) + '. App cần đúng thứ tự cột như file gốc.')
def load_raw(data, kt, master, ads=None):
    """data/kt/master/ads: đường dẫn hoặc bytes. Báo ValueError tiếng Việt nếu file sai sheet/cột."""
    xd = _open_xl(data, 'TUYỂN DỤNG - DATA', ['DATA TUYỂN DỤNG'])
    d = pd.read_excel(xd, sheet_name='DATA TUYỂN DỤNG', header=0)
    if d.shape[1] < 34: raise ValueError(f"Sheet 'DATA TUYỂN DỤNG' chỉ có {d.shape[1]} cột (cần ít nhất 34). Có phải bạn chọn nhầm file?")
    d = d.iloc[:, :34]
    _check_headers(list(d.columns), {1: 'KHOI', 2: 'CHI NHANH', 3: 'VI TRI', 4: 'TEN', 8: 'NGUON', 9: 'DANH GIA', 11: 'NGAY PV', 15: 'LIEN LAC', 17: 'KET QUA', 20: 'NHAN VIEC', 21: 'KET QUA NHAN VIEC', 23: 'LUONG', 31: 'NGHI VIEC'}, "sheet DATA TUYỂN DỤNG")
    d.columns = DATA_COLS
    d = d.dropna(subset=['TEN']).copy()
    if not len(d): raise ValueError("Sheet 'DATA TUYỂN DỤNG' không có dòng dữ liệu nào.")
    dq = {}
    for c in ['NGAYPV', 'NGAYLL', 'NGAYNV', 'NGAYNGHI']:
        d[c + '_d'] = d[c].map(pdate)
        dq['text_' + c] = int((d[c].map(lambda v: isinstance(v, str)) & d[c + '_d'].notna()).sum())   # ô ngày nhập dạng chữ đã đọc được
    for c in ['KHOI', 'CN', 'VITRI', 'NGUON', 'LH', 'KQPV', 'KQNV', 'TTLV', 'DANHGIA', 'NGUOIPV', 'LUONG']: d[c] = txt(d[c])
    d['NGUON'] = d.NGUON.fillna('(không ghi nguồn)'); d['LH'] = d.LH.fillna('(không ghi người LH)')
    d['DON_VI'] = d.CN.fillna('(không ghi)'); d['k'] = d.TEN.map(norm)
    xk = _open_xl(kt, 'KẾ TOÁN ADMIN UPDATE', ['CHECK LIST NGHỈ VIỆC', 'NGHỈ VIỆC', 'NHẬN VIỆC'])
    ck = pd.read_excel(xk, sheet_name='CHECK LIST NGHỈ VIỆC')
    _check_headers(list(ck.columns), {0: 'CHI NHANH', 3: 'HO TEN', 7: 'NGAY NHAN VIEC', 9: 'NGAY NGHI VIEC'}, 'sheet CHECK LIST NGHỈ VIỆC (Kế toán)')
    ck = ck.dropna(subset=[ck.columns[3]]).iloc[:, :11]
    ck.columns = ['CN', 'TT', 'MSNV', 'TEN', 'BP', 'CV', 'CD', 'NNV', 'NCTV', 'NNGHI', 'LYDO']
    kq = pd.DataFrame(kt_block(kt, 'NGHỈ VIỆC', 9), columns=['CN', 'TT', 'TEN', 'CV', 'NNV', 'NNGHI', 'x', 'CRM', 'NLV', 'ROW'])
    kn = pd.DataFrame(kt_block(kt, 'NHẬN VIỆC', 6), columns=['CN', 'TT', 'TEN', 'CV', 'NNV', 'CRM', 'ROW'])
    xm = _open_xl(master, 'MASTER DATA', ['1. TỔNG DATA - NHÂN SỰ'])
    ms = pd.read_excel(xm, sheet_name='1. TỔNG DATA - NHÂN SỰ', usecols=range(0, 12))
    _check_headers(list(ms.columns), {0: 'CHI NHANH', 3: 'HO TEN', 7: 'NGAY NHAN VIEC', 9: 'NGAY NGHI VIEC'}, 'sheet 1. TỔNG DATA - NHÂN SỰ (Master)')
    ms.columns = ['CN', 'TT', 'MSNV', 'TEN', 'BP', 'CV', 'CD', 'NNV', 'NCTV', 'NNGHI', 'LYDO', 'NDC']; ms = ms.dropna(subset=['TEN']).copy()
    for df in (ck, kq, kn, ms):
        df['NNV_d'] = df.NNV.map(pdate); df['k'] = df.TEN.map(norm)
        if 'NNGHI' in df: df['NNGHI_d'] = df.NNGHI.map(pdate)
    a = None
    if ads is not None:
        a = pd.read_excel(_b(ads)); a = a.iloc[:, :6]; a.columns = ['ngay', 'ad', 'cur', 'spend', 'msg', 'cpm']
        a['ngay'] = pd.to_datetime(a.ngay, errors='coerce'); a['spend'] = pd.to_numeric(a.spend, errors='coerce').fillna(0); a['msg'] = pd.to_numeric(a.msg, errors='coerce').fillna(0)
        a = a.dropna(subset=['ngay'])
    dq['master_nghi_khong_ngay'] = int((ms.TT.isin(['Đã nghỉ việc', 'Nghỉ ngang', 'Công ty cho nghỉ']) & ms.NNGHI_d.isna()).sum())
    dq['data_thieu_ngay_lien_lac'] = int(d.NGAYLL_d.isna().sum())
    pairs = []
    try:
        wsb = openpyxl.load_workbook(_b(data), read_only=True, data_only=True)['BC-2026']
        for b_, c_ in wsb.iter_rows(min_row=34, max_row=245, min_col=2, max_col=3, values_only=True):
            if b_ is None or c_ is None or str(b_).strip().upper() == 'TỔNG': continue
            pairs.append((str(b_), str(c_)))
    except Exception: pairs = []
    raw = dict(d=d, ck=ck, kq=kq, kn=kn, ms=ms, ads=a, dq=dq, pairs=pairs)
    raw['nghi_all'] = build_nghi(raw)
    cands = [d.NGAYLL_d.max(), ms.NNV_d.max(), ms.NNGHI_d.max(), ck.NNGHI_d.max(), kq.NNGHI_d.max()]
    today = pd.Timestamp.today().normalize()   # bỏ qua ngày dự kiến trong tương lai (vd. ngày nghỉ đã báo trước)
    raw['cutoff'] = min(max([c for c in cands if pd.notna(c)] or [today]), today)
    return raw

def build_nghi(raw):
    """Hợp mọi người có ngày nghỉ ở Master / CHECK LIST / NGHỈ VIỆC (gộp theo tên + ngày nhận việc ±7 ngày)."""
    pers = {}
    out = []
    for src, df in (('Master', raw['ms']), ('CHECK LIST', raw['ck']), ('KT NGHỈ VIỆC', raw['kq'])):
        for r in df[df.NNGHI_d.notna()].itertuples(index=False):
            k = r.k; nnv = r.NNV_d; hit = None
            for p in pers.get(k, []):
                if pd.isna(nnv) or pd.isna(p['NNV']) or abs((nnv - p['NNV']).days) <= 7:
                    if abs((r.NNGHI_d - p['NNGHI']).days) <= 40: hit = p; break
            g = lambda n: getattr(r, n, None)
            if hit:
                hit['src'].add(src); hit['dates'].add(r.NNGHI_d)
                for kk, vv in (('MSNV', g('MSNV')), ('BP', g('BP')), ('CV', g('CV')), ('LYDO', g('LYDO')), ('TT', g('TT'))):
                    if (hit.get(kk) is None or (isinstance(hit.get(kk), float) and pd.isna(hit.get(kk)))) and vv is not None and not (isinstance(vv, float) and pd.isna(vv)): hit[kk] = vv
                if pd.isna(hit['NNV']): hit['NNV'] = nnv
                continue
            p = dict(k=k, TEN=r.TEN, CN=r.CN, NNV=nnv, NNGHI=r.NNGHI_d, MSNV=g('MSNV'), TT=g('TT'), BP=g('BP'), CV=g('CV'), LYDO=g('LYDO'), src={src}, dates={r.NNGHI_d})
            pers.setdefault(k, []).append(p); out.append(p)
    n = pd.DataFrame(out)
    if n.empty: return n
    n['LECH_NGAY'] = n.dates.map(lambda x: len(x) > 1); n['LECH_THANG'] = n.dates.map(lambda x: len({(t.year, t.month) for t in x}) > 1)
    n = n.drop(columns=['dates'])
    n['src'] = n.src.map(lambda s: ', '.join(sorted(s)))
    n['days'] = (n.NNGHI - n.NNV).dt.days
    for c in ['CN', 'TT', 'BP', 'CV', 'LYDO']: n[c] = txt(n[c].astype(object))
    n['CN'] = n.CN.fillna('(không ghi)')
    n['NHOM'] = n.days.map(lambda x: bucket(x) if pd.notna(x) else None)
    return n

# ------------------------------------------------------------------ phân tích tuyển dụng
def agg_by(d, key, s, e, delta=None):
    g = d.assign(_k=d[key])
    cv = g[_in(g.NGAYLL_d, s, e)].groupby('_k').size()
    pvm = _in(g.NGAYPV_d, s, e)
    pv = g[pvm & g.KQPV.isin(PVOK)].groupby('_k').size(); ps = g[pvm & (g.KQPV == 'Đạt')].groupby('_k').size()
    kd = g[pvm & (g.KQPV == 'Không đạt')].groupby('_k').size(); ns = g[pvm & (g.KQPV == 'Không đến pv')].groupby('_k').size()
    nvm = _in(g.NGAYNV_d, s, e)
    nv = g[nvm & (g.KQNV == 'Đã nhận việc')].groupby('_k').size(); knv = g[nvm & (g.KQNV == 'Không nhận việc')].groupby('_k').size()
    t = pd.concat({'CV': cv, 'PV': pv, 'PASS': ps, 'KHONG_DAT': kd, 'KHONG_DEN_PV': ns, 'NV': nv, 'KHONG_NV': knv}, axis=1).fillna(0).astype(int)
    t.index.name = key
    # đợt CV tháng này: kết quả cuối cùng của đúng những người liên lạc trong tháng
    c = g[_in(g.NGAYLL_d, s, e)]
    co = pd.concat({'CV_DOT': c.groupby('_k').size(), 'DOT_PV': c[c.KQPV.isin(PVOK)].groupby('_k').size(), 'DOT_PASS': c[c.KQPV == 'Đạt'].groupby('_k').size(), 'DOT_NV': c[c.KQNV == 'Đã nhận việc'].groupby('_k').size()}, axis=1).fillna(0).astype(int)
    t = t.join(co, how='outer').fillna(0).astype(int)
    for (g_, mm_), v_ in (delta or {}).items():
        if g_ not in t.index: t.loc[g_] = 0
        t.loc[g_, mm_] = int(t.loc[g_, mm_]) + int(v_)
    t = t.astype(int); t.index.name = key
    t['%PV/CV'] = [pct(a, b) for a, b in zip(t.PV, t.CV)]
    t['%Pass/PV'] = [pct(a, b) for a, b in zip(t.PASS, t.PV)]
    t['%NV/Pass'] = [pct(a, b) for a, b in zip(t.NV, t.PASS)]
    t['%NV/CV'] = [pct(a, b) for a, b in zip(t.NV, t.CV)]
    t['%Không đến PV'] = [pct(a, b) for a, b in zip(t.KHONG_DEN_PV, t.PV + t.KHONG_DEN_PV)]
    t['%Đợt CV→PV'] = [pct(a, b) for a, b in zip(t.DOT_PV, t.CV_DOT)]
    t['%Đợt CV→Pass'] = [pct(a, b) for a, b in zip(t.DOT_PASS, t.CV_DOT)]
    t['%Đợt CV→NV'] = [pct(a, b) for a, b in zip(t.DOT_NV, t.CV_DOT)]
    tot = t[['CV', 'PV', 'PASS', 'KHONG_DAT', 'KHONG_DEN_PV', 'NV', 'KHONG_NV', 'CV_DOT', 'DOT_PV', 'DOT_PASS', 'DOT_NV']].sum()
    t['%Tỷ trọng NV'] = [pct(a, tot['NV']) for a in t.NV]
    t['%Tỷ trọng CV'] = [pct(a, tot['CV']) for a in t.CV]
    t = t[(t[['CV', 'PV', 'PASS', 'NV', 'KHONG_NV', 'KHONG_DEN_PV']].sum(axis=1) > 0)]
    return t.sort_values(['NV', 'CV'], ascending=False)

REASON_RULES = [('Gia đình / cá nhân / về quê', ['gia dinh', 'ca nhan', 've que', 'dam cuoi', 'co con', 'mang thai', 'sinh', 'chong', 'vo ', 'ba ', 'me ']),
    ('Sức khỏe', ['suc khoe', 'benh', 'dau', 'om ']), ('Công việc không phù hợp / áp lực KPI', ['cong viec', 'ap luc', 'kpi', 'doanh so', 'khong phu hop', 'ko phu hop', 'chua phu hop']),
    ('Môi trường / đồng nghiệp / quản lý', ['moi truong', 'dong nghiep', 'quan ly', 'khong hop']), ('Khoảng cách / đi lại / chi nhánh', ['khoang cach', 'di lai', 'xa ', 'chi nhanh', 'dieu chuyen']),
    ('Học tập / thi cử', ['hoc', 'thi ']), ('Công ty cho nghỉ / không đạt', ['cty cho nghi', 'cong ty cho nghi', 'khong dat', 'cho thoi viec', 'cpl']),
    ('Bỏ việc / không liên hệ được', ['bo viec', 'nghi ngang', 'khong lien he', 'khong lien lac', 'ko lien'])]
def reason_group(t):
    if t is None or (isinstance(t, float) and pd.isna(t)) or not str(t).strip(): return '(không ghi lý do)'
    n = ' ' + norm(t).lower() + ' '
    for name, kws in REASON_RULES:
        if any(k in n for k in kws): return name
    return 'Khác'

BRANCHES = ['QUẬN 1', 'PHÚ NHUẬN', 'TÂN BÌNH', 'GÒ VẤP', 'QUẬN 10', 'TÂN PHÚ', 'BÌNH TÂN', 'VŨNG TÀU', 'BÌNH DƯƠNG', 'THỦ ĐỨC']
DEPTS = ['MKT', 'SO', 'HCNS', 'KẾ TOÁN', 'VẬN HÀNH', 'DỰ ÁN']
DEPT_METRICS = ['Số CV', 'Tham gia PV', 'Đạt PV', 'Không đạt PV', 'Đã NV', 'Không nhận việc']

def dept_table(raw, s, e, delta=None):
    """Bảng phòng ban / chi nhánh đúng định nghĩa báo cáo: chỉ tính các (chi nhánh, vị trí) có trong danh sách của sheet BC-2026."""
    pairs = raw.get('pairs') or []
    if not pairs: return pd.DataFrame()
    d = raw['d']; low_cn = d.DON_VI.fillna('').str.lower(); low_vt = d.VITRI.fillna('').str.lower(); acc = {}
    for b_, c_ in pairs:
        x = d[(low_cn == b_.lower()) & (low_vt == c_.lower())]
        v = dict(CV=int(_in(x.NGAYLL_d, s, e).sum()), PV=int((_in(x.NGAYPV_d, s, e) & x.KQPV.isin(PVOK)).sum()), DAT=int((_in(x.NGAYPV_d, s, e) & (x.KQPV == 'Đạt')).sum()),
                 KDAT=int((_in(x.NGAYPV_d, s, e) & (x.KQPV == 'Không đạt')).sum()), NV=int((_in(x.NGAYNV_d, s, e) & (x.KQNV == 'Đã nhận việc')).sum()), KNV=int((_in(x.NGAYNV_d, s, e) & (x.KQNV == 'Không nhận việc')).sum()))
        g_ = b_.upper(); acc.setdefault(g_, dict(CV=0, PV=0, DAT=0, KDAT=0, NV=0, KNV=0))
        for k_, vv in v.items(): acc[g_][k_] += vv
    for (g_, k_), vv in (delta or {}).items(): acc.setdefault(g_, dict(CV=0, PV=0, DAT=0, KDAT=0, NV=0, KNV=0))[k_] += vv
    z = dict(CV=0, PV=0, DAT=0, KDAT=0, NV=0, KNV=0)
    t = pd.DataFrame({g_: acc.get(g_, z) for g_ in DEPTS + BRANCHES}); t.index = DEPT_METRICS
    t['TỔNG PHÒNG BAN'] = t[DEPTS].sum(axis=1); t['TỔNG 10 CHI NHÁNH'] = t[BRANCHES].sum(axis=1)
    return t

def dat_thu_viec(raw, s, e, adj=None):
    """Người hết thử việc trong tháng và còn làm việc (theo ngày cuối thử việc của Master)."""
    adj = adj or {}; ms = raw['ms'].copy(); ms['NCTV_d'] = ms.NCTV.map(pdate)
    extra = set((adj.get('dat_thu_viec_bo_sung') or {}).keys())
    dt = ms[_in(ms.NCTV_d, s, e) & ((ms.TT == 'Đang làm việc') | ms.MSNV.isin(extra))].copy()
    ov = adj.get('cn_ghi_de') or {}
    dt['CN_BC'] = dt.apply(lambda x: ov.get(norm(x.TEN), x.CN), axis=1)
    dt['NHOM'] = dt.CN_BC.map(lambda v: 'VĂN PHÒNG' if norm(v) == 'VAN PHONG' else str(v).strip().upper())
    cnt = {'VĂN PHÒNG': int((dt.NHOM == 'VĂN PHÒNG').sum())}
    for b_ in BRANCHES: cnt[b_] = int((dt.NHOM.map(norm) == norm(b_)).sum())
    ser = pd.Series(cnt, name='Số người'); ser['TỔNG'] = int(ser.sum())
    return ser, dt

def branch_table(R, dau_ky=None, sl_yc=None):
    """Bảng biến động nhân sự theo chi nhánh đúng bố cục báo cáo (Đầu kỳ / SL yêu cầu nhập tay)."""
    nv = R['nv_list']; ng = R['nghi_list']; rows = []
    for b_ in BRANCHES:
        t = int(((nv.CN.map(norm) == norm(b_)) & (nv.KHOI != 'Văn Phòng')).sum()) if len(nv) else 0
        m_ = ng[ng.CN.map(norm) == norm(b_)] if len(ng) else ng
        ct = int((m_.days >= 30).sum()) if len(m_) else 0; tv = int((m_.days < 30).sum()) if len(m_) else 0
        dk = (dau_ky or {}).get(b_); sl = (sl_yc or {}).get(b_)
        rows.append({'Chi nhánh': b_, 'Đầu kỳ': dk, 'SL yêu cầu': sl, 'Tuyển mới': t, 'Nghỉ chính thức': ct, 'Nghỉ thử việc': tv, 'Tổng nghỉ': ct + tv,
                     'Cuối kỳ': (dk + t - ct - tv) if dk is not None else np.nan})
    df = pd.DataFrame(rows).set_index('Chi nhánh'); df['Net HC'] = df['Cuối kỳ'] - df['Đầu kỳ']
    df['% Tuyển mới / SL yêu cầu'] = [pct(a, b) if (b is not None and b == b and b > 0) else np.nan for a, b in zip(df['Tuyển mới'], df['SL yêu cầu'])]
    df['% Nghỉ / Đầu kỳ'] = [pct(a, b) if (b is not None and b == b and b > 0) else np.nan for a, b in zip(df['Tổng nghỉ'], df['Đầu kỳ'])]
    tot = df.drop(columns=['% Tuyển mới / SL yêu cầu', '% Nghỉ / Đầu kỳ', 'Net HC']).sum(numeric_only=True); tot.name = 'TỔNG'
    out = pd.concat([df, tot.to_frame().T]); out.loc['TỔNG', 'Net HC'] = out.loc['TỔNG', 'Cuối kỳ'] - out.loc['TỔNG', 'Đầu kỳ']
    out.loc['TỔNG', '% Tuyển mới / SL yêu cầu'] = pct(out.loc['TỔNG', 'Tuyển mới'], out.loc['TỔNG', 'SL yêu cầu']); out.loc['TỔNG', '% Nghỉ / Đầu kỳ'] = pct(out.loc['TỔNG', 'Tổng nghỉ'], out.loc['TỔNG', 'Đầu kỳ'])
    out.index.name = 'Chi nhánh'
    return out

def compute_month(raw, y, m, cost_manual=None, adjust=None, period=None, label=None):
    """adjust: khối điều chỉnh (dieu_chinh.get(y, m)) để khớp báo cáo đã duyệt; None = số thuần từ raw."""
    s, e = (pd.Timestamp(period[0]), pd.Timestamp(period[1])) if period else month_range(y, m); d = raw['d']; nga = raw['nghi_all']; adj = adjust or {}
    R = dict(year=int(y), month=int(m), start=s, end=e, adjusted=bool(adjust), adj=adj, label=label or f'Tháng {int(m):02d}/{int(y)}', is_month=(period is None))
    in_cv = _in(d.NGAYLL_d, s, e); pvm = _in(d.NGAYPV_d, s, e); nvm = _in(d.NGAYNV_d, s, e)
    T = dict(CV=int(in_cv.sum()), PV=int((pvm & d.KQPV.isin(PVOK)).sum()), PASS=int((pvm & (d.KQPV == 'Đạt')).sum()),
             KHONG_DAT=int((pvm & (d.KQPV == 'Không đạt')).sum()), DOI_KQ=int((pvm & (d.KQPV == 'Đợi kết quả')).sum()), KHONG_DEN_PV=int((pvm & (d.KQPV == 'Không đến pv')).sum()),
             NV=int((nvm & (d.KQNV == 'Đã nhận việc')).sum()), KHONG_NV=int((nvm & (d.KQNV == 'Không nhận việc')).sum()))
    R['funnel_raw'] = dict(T)
    dh = adj.get('delta_hr') or {}
    for k_, m_ in (('PV', 'PV'), ('PASS', 'PASS'), ('NV', 'NV')): T[k_] += sum(v for (g_, mm), v in dh.items() if mm == m_)
    T['KHONG_NV'] += sum(v for (g_, mm), v in (adj.get('delta_nguon') or {}).items() if mm == 'KHONG_NV')
    R['funnel'] = T
    R['has_data'] = T['CV'] > 0 or T['NV'] > 0
    R['by_source'] = agg_by(d, 'NGUON', s, e, adj.get('delta_nguon')); R['by_hr'] = agg_by(d, 'LH', s, e, adj.get('delta_hr')); R['by_unit'] = agg_by(d, 'DON_VI', s, e, adj.get('delta_unit')); R['by_pos'] = agg_by(d, 'VITRI', s, e)
    kh = d.assign(KHOI=d.KHOI.fillna('(không ghi)')); R['by_khoi'] = agg_by(kh, 'KHOI', s, e)
    R['dept_report'] = dept_table(raw, s, e, adj.get('delta_dept'))
    # ---- nhận việc chi tiết + giữ chân
    nv = d[nvm & (d.KQNV == 'Đã nhận việc')][['TEN', 'k', 'KHOI', 'CN', 'VITRI', 'NGUON', 'LH', 'NGAYNV_d']].rename(columns={'NGAYNV_d': 'NNV'}).copy()
    qd = []
    for r in nv.itertuples(index=False):
        c = nga[(nga.k == r.k) & ((nga.NNV - r.NNV).abs() <= pd.Timedelta(days=7))] if len(nga) else nga
        qd.append(int((c.iloc[0].NNGHI - r.NNV).days) if len(c) else np.nan)
    nv['NGAY_DEN_KHI_NGHI'] = qd; nv['NGHI_CHO_BAO_CAO'] = ''
    tm = adj.get('ten_theo_bao_cao') or {}
    if tm: nv['TEN'] = nv.TEN.map(lambda t: tm.get(norm(t), t))
    sup = pd.DataFrame([dict(TEN=x['TEN'], k=norm(x['TEN']), KHOI=x['KHOI'], CN=x['CN'], VITRI=x['VITRI'], NGUON=x['NGUON'], LH=x['LH'], NNV=pd.Timestamp(x['NNV']), NGAY_DEN_KHI_NGHI=np.nan, NGHI_CHO_BAO_CAO='Bổ sung theo báo cáo: ' + x['ly_do']) for x in adj.get('nhan_viec_bo_sung', [])])
    if len(sup): nv = pd.concat([nv, sup], ignore_index=True)
    nv['DA_NGHI'] = nv.NGAY_DEN_KHI_NGHI.notna(); nv = nv.rename(columns={'NGHI_CHO_BAO_CAO': 'GHI_CHU'})
    R['nv_list'] = nv
    cut = raw['cutoff']; R['cutoff'] = cut
    R['retention'] = dict(n=len(nv), nghi30=int((nv.NGAY_DEN_KHI_NGHI <= 30).sum()), nghi60=int((nv.NGAY_DEN_KHI_NGHI <= 60).sum()), nghi_all=int(nv.DA_NGHI.sum()),
                          du_30=(e + pd.Timedelta(days=30) <= cut), du_60=(e + pd.Timedelta(days=60) <= cut))
    if len(nv):
        rs = nv.groupby('NGUON').agg(NV=('TEN', 'size'), NGHI=('DA_NGHI', 'sum'), NGHI30=('NGAY_DEN_KHI_NGHI', lambda x: int((x <= 30).sum()))).reset_index()
        rs['%Đã nghỉ'] = [pct(a, b) for a, b in zip(rs.NGHI, rs.NV)]; rs['%Nghỉ ≤30 ngày'] = [pct(a, b) for a, b in zip(rs.NGHI30, rs.NV)]
        R['retention_src'] = rs.sort_values('NV', ascending=False)
    else: R['retention_src'] = pd.DataFrame(columns=['NGUON', 'NV', 'NGHI', 'NGHI30', '%Đã nghỉ', '%Nghỉ ≤30 ngày'])
    # ---- nghỉ việc
    ng = nga[_in(nga.NNGHI, s, e)].copy() if len(nga) else nga.copy()
    R['nghi_excluded'] = ng.iloc[0:0]; R['nghi_total_raw'] = len(ng)
    lt = {(k_, dt_): why for k_, dt_, why in adj.get('loai_tru_nghi', [])}
    if len(ng) and lt:
        mk = ng.apply(lambda r: (r.k, str(r.NNV.date())) in lt if pd.notna(r.NNV) else False, axis=1)
        R['nghi_excluded'] = ng[mk].assign(LY_DO_LOAI=[lt[(r.k, str(r.NNV.date()))] for r in ng[mk].itertuples(index=False)]); ng = ng[~mk].copy()
    if len(ng):
        src = []; lh = []
        for r in ng.itertuples(index=False):
            c = d[(d.k == r.k) & (d.KQNV == 'Đã nhận việc')]
            if pd.notna(r.NNV) and len(c): c = c[(c.NGAYNV_d - r.NNV).abs() <= pd.Timedelta(days=7)]
            src.append(c.iloc[0].NGUON if len(c) else '(không rõ)'); lh.append(c.iloc[0].LH if len(c) else '(không rõ)')
        ng['NGUON'] = src; ng['LH'] = lh; ng['LOAI'] = ng.days.map(lambda x: 'Nghỉ thử việc (<30 ngày)' if pd.notna(x) and x < 30 else ('Nghỉ chính thức (≥30 ngày)' if pd.notna(x) else '(thiếu ngày)'))
        ng['NHOM_LY_DO'] = ng.LYDO.map(reason_group); ng['NHAN_VIEC_CUNG_THANG'] = (ng.NNV >= s)
    R['nghi_list'] = ng
    nT = len(ng); R['nghi_total'] = nT
    if nT:
        t = ng.NHOM.value_counts().reindex(BUCKETS).fillna(0).astype(int).to_frame('So_nguoi'); t['%'] = [pct(a, nT) for a in t.So_nguoi]; t['% cộng dồn'] = t['%'].cumsum(); R['tenure'] = t
        u = ng.groupby('CN').agg(NGHI=('TEN', 'size'), NGHI_TV=('days', lambda x: int((x < 30).sum()))).sort_values('NGHI', ascending=False); u['%Tỷ trọng'] = [pct(a, nT) for a in u.NGHI]; u['%Nghỉ thử việc'] = [pct(a, b) for a, b in zip(u.NGHI_TV, u.NGHI)]; R['nghi_unit'] = u
        v = ng.groupby(ng.CV.fillna('(không ghi)')).size().sort_values(ascending=False).to_frame('NGHI'); v['%'] = [pct(a, nT) for a in v.NGHI]; R['nghi_pos'] = v
        if adj.get('ly_do_nhom'):
            w = pd.Series(adj['ly_do_nhom']).to_frame('So_nguoi'); w['%'] = [pct(a, nT) for a in w.So_nguoi]; R['reason_manual'] = True
        else:
            w = ng.NHOM_LY_DO.value_counts().to_frame('So_nguoi'); w['%'] = [pct(a, nT) for a in w.So_nguoi]; R['reason_manual'] = False
        R['nghi_reason'] = w
        sx = ng.groupby('NGUON').size().sort_values(ascending=False).to_frame('NGHI'); sx['%'] = [pct(a, nT) for a in sx.NGHI]; R['nghi_src'] = sx
        R['nghi_tv'] = int((ng.days < 30).sum()); R['nghi_early10'] = int((ng.days <= 10).sum()); R['nghi_moi'] = int(ng.NHAN_VIEC_CUNG_THANG.sum())
    else:
        for k in ('tenure', 'nghi_unit', 'nghi_pos', 'nghi_reason', 'nghi_src'): R[k] = pd.DataFrame()
        R['nghi_tv'] = R['nghi_early10'] = R['nghi_moi'] = 0; R['reason_manual'] = False
    R['dat_thu_viec'], R['dat_thu_viec_list'] = dat_thu_viec(raw, s, e, adj)
    # ---- độ đầy đủ dữ liệu giữa các file
    ms = raw['ms']; kn = raw['kn']
    cov = []
    for r in nv.itertuples(index=False):
        cm = ms[(ms.k == r.k) & ((ms.NNV_d - r.NNV).abs() <= pd.Timedelta(days=7))]; ck_ = kn[(kn.k == r.k) & ((kn.NNV_d - r.NNV).abs() <= pd.Timedelta(days=7))]
        cov.append((len(cm) > 0, len(ck_) > 0 or r.KHOI == 'Văn Phòng'))
    xx = d[in_cv]; R['dup_cv'] = int(xx.duplicated(['TEN', 'VITRI', 'NGAYLL_d', 'NGUON'], keep='first').sum())
    R['coverage'] = dict(nv_data=len(nv), nv_in_master=sum(a for a, b in cov), nv_in_kt=sum(b for a, b in cov), master_nv_month=int(_in(ms.NNV_d, s, e).sum()), kt_nv_month=int(_in(kn.NNV_d, s, e).sum()),
                         nghi_lech_ngay=int(ng.LECH_NGAY.sum()) if nT else 0, nghi_lech_thang=int(ng.LECH_THANG.sum()) if nT else 0, nghi_n=nT, nghi_in_master=int(ng.src.str.contains('Master').sum()) if nT else 0, nghi_in_ck=int(ng.src.str.contains('CHECK LIST').sum()) if nT else 0, nghi_in_kq=int(ng.src.str.contains('KT NGHỈ VIỆC').sum()) if nT else 0)
    # ---- chi phí
    cm_ = dict(cost_manual or {}); a = raw['ads']
    ads = a[_in(a.ngay, s, e)].groupby('ngay').agg(spend=('spend', 'sum'), msg=('msg', 'sum')).reset_index() if a is not None else pd.DataFrame(columns=['ngay', 'spend', 'msg'])
    extra_days = []
    for dd, (sp, ms_) in (adj.get('ads_bo_sung') or {}).items():
        dd = pd.Timestamp(dd)
        if _in(pd.Series([dd]), s, e).iloc[0] and dd not in set(ads.ngay): ads = pd.concat([ads, pd.DataFrame([dict(ngay=dd, spend=float(sp), msg=float(ms_))])], ignore_index=True); extra_days.append(dd)
    ads = ads.sort_values('ngay'); fb = float(ads.spend.sum()) if len(ads) else 0.0; msg = float(ads.msg.sum()) if len(ads) else 0.0
    other = float(sum(cm_.values())); tc = fb + other
    R['cost'] = dict(fb=fb, msg=msg, ads_days=int(ads.ngay.nunique()), ads_extra_days=len(extra_days), days_in_month=int((e - s).days + 1), manual=cm_, total=tc, cpm=(fb / msg if msg else np.nan),
                     per_cv=(tc / T['CV'] if T['CV'] and tc else np.nan), per_pv=(tc / T['PV'] if T['PV'] and tc else np.nan), per_pass=(tc / T['PASS'] if T['PASS'] and tc else np.nan), per_nv=(tc / T['NV'] if T['NV'] and tc else np.nan))
    R['ads_daily'] = ads
    bl = adj.get('baseline_raw'); R['adj_baseline_diff'] = {}
    if bl:
        for k_, v_ in bl['funnel'].items():
            if R['funnel_raw'][k_] != v_: R['adj_baseline_diff'][k_] = (R['funnel_raw'][k_], v_)
        if R['nghi_total_raw'] != bl['nghi_raw']: R['adj_baseline_diff']['NGHI'] = (R['nghi_total_raw'], bl['nghi_raw'])
    return R

def kpi_row(R):
    T = R['funnel']; c = R['cost']
    return {'Số CV (liên lạc)': T['CV'], 'Tham gia PV': T['PV'], 'Đạt PV (Pass)': T['PASS'], 'Không đạt PV': T['KHONG_DAT'], 'Không đến PV': T['KHONG_DEN_PV'], 'Nhận việc': T['NV'], 'Không nhận việc': T['KHONG_NV'],
            '% PV / CV': pct(T['PV'], T['CV']), '% Pass / PV': pct(T['PASS'], T['PV']), '% Nhận việc / Pass': pct(T['NV'], T['PASS']), '% Nhận việc / CV': pct(T['NV'], T['CV']),
            'Nghỉ việc': R['nghi_total'], '  – nghỉ thử việc (<30 ngày)': R['nghi_tv'], '% nghỉ thử việc / nghỉ việc': pct(R['nghi_tv'], R['nghi_total']), '% nghỉ việc / nhận việc': pct(R['nghi_total'], T['NV']),
            'Tổng chi phí (VND)': c['total'], 'Chi phí / CV': c['per_cv'], 'Chi phí / người nhận việc': c['per_nv']}
def compare(results):
    cols = {f"T{R['month']:02d}/{R['year']}": kpi_row(R) for R in results}
    df = pd.DataFrame(cols)
    names = list(df.columns)
    for i in range(1, len(names)):
        a, b = df[names[i - 1]], df[names[i]]
        df[f'Δ% {names[i-1]}→{names[i]}'] = [((y - x) / abs(x) * 100) if (pd.notna(x) and pd.notna(y) and x not in (0,)) and not str(idx).startswith('%') else np.nan for idx, x, y in zip(df.index, a, b)]
    return df


XL_NOTES = [
    'CV = ngày liên lạc trong tháng. Tham gia PV = ngày PV trong tháng & KQ ∈ {Đạt, Đợi kết quả, Không đạt}. Pass = ngày PV trong tháng & Đạt. Nhận việc = ngày nhận việc trong tháng & "Đã nhận việc".',
    'Nghỉ việc = hợp CHECK LIST + sheet NGHỈ VIỆC (Kế toán) + Master theo ngày nghỉ (ưu tiên Master → CHECK LIST → NGHỈ VIỆC khi ngày khác nhau). Nghỉ thử việc = làm dưới 30 ngày.',
    'Tỷ lệ "theo kỳ" tính mỗi bước theo ngày của bước đó nên có thể vượt 100%; tỷ lệ "theo đợt CV" chỉ xét người liên lạc trong tháng nên luôn ≤100%.',
    'Phân tích chuyên sâu (mục 12) tính từ dòng raw của tháng, không gồm điều chỉnh khớp báo cáo.',
    'Nhóm lý do nghỉ: tháng có điều chỉnh dùng 12 nhóm phân loại tay của báo cáo; tháng khác gán tự động theo từ khóa, chỉ tham khảo.',
    'Người chưa có trong file nghỉ việc được coi là còn làm khi tính giữ chân; chỉ tính người đã đủ số ngày tới ngày dữ liệu mới nhất.']

def _pct_col(name): n = str(name); return n.startswith('%') or n.startswith('Δ%') or n.startswith('% ')
def _num_kind(v): return 'int' if (isinstance(v, (int, np.integer)) or (isinstance(v, (float, np.floating)) and float(v).is_integer())) else 'flt'
def _write_df(ws, df, row, F, index=True, widths=None):
    """Ghi DataFrame vào sheet bắt đầu từ dòng `row` (0-based), trả về dòng tiếp theo."""
    if isinstance(df, pd.Series): df = df.to_frame()
    if index and not isinstance(df.index, pd.RangeIndex):
        df = df.copy(); df.index.name = df.index.name or 'Chỉ tiêu'; df = df.reset_index()
    cols = list(df.columns)
    for j, c in enumerate(cols): ws.write(row, j, str(c), F['hdr'])
    for i, rec in enumerate(df.itertuples(index=False), start=row + 1):
        for j, v in enumerate(rec):
            c = cols[j]
            if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NaT: ws.write_blank(i, j, None, F['txt']); continue
            if isinstance(v, (pd.Timestamp, datetime.datetime, datetime.date)): ws.write_datetime(i, j, v.to_pydatetime() if isinstance(v, pd.Timestamp) else v, F['date']); continue
            if isinstance(v, (bool, np.bool_)): ws.write_string(i, j, 'Có' if v else 'Không', F['txt']); continue
            if isinstance(v, (int, float, np.integer, np.floating)):
                ws.write_number(i, j, float(v), F['pct'] if _pct_col(c) else F[_num_kind(v)]); continue
            ws.write_string(i, j, str(v), F['txt'])
        if widths is not None:
            for j, v in enumerate(rec): widths[j] = max(widths.get(j, 0), min(len(str(v)) if v is not None else 0, 45))
    return row + len(df) + 2

def month_sections(R, Dp, bt):
    """Danh sách (tiêu đề, bảng) của một tháng – dùng chung cho Excel và kiểm thử."""
    T = R['funnel']; c = R['cost']; S = []
    S.append(('1. Chỉ tiêu chính', pd.DataFrame({'Giá trị': pd.Series(kpi_row(R))})))
    fun = pd.DataFrame({'Bước': ['CV (liên lạc)', 'Tham gia PV', 'Đạt PV', 'Nhận việc'], 'Số lượng': [T['CV'], T['PV'], T['PASS'], T['NV']]})
    fun['% so với bước trước'] = [np.nan] + [pct(fun['Số lượng'][i], fun['Số lượng'][i - 1]) for i in range(1, 4)]; fun['% so với CV'] = [pct(x, T['CV']) for x in fun['Số lượng']]
    S.append(('2. Phễu tuyển dụng', fun.set_index('Bước')))
    S.append(('2b. Kết quả phỏng vấn / nhận việc', pd.DataFrame({'Số lượng': [T['KHONG_DAT'], T['KHONG_DEN_PV'], T['DOI_KQ'], T['KHONG_NV']],
                '%': [pct(T['KHONG_DAT'], T['PV']), pct(T['KHONG_DEN_PV'], T['PV'] + T['KHONG_DEN_PV']), pct(T['DOI_KQ'], T['PV']), pct(T['KHONG_NV'], T['NV'] + T['KHONG_NV'])]},
                index=pd.Index(['Không đạt PV (% của tham gia PV)', 'Không đến PV (% của số hẹn)', 'Đợi kết quả (% của tham gia PV)', 'Không nhận việc (% của số được mời)'], name='Chỉ tiêu'))))
    kyc = ['CV', 'PV', 'PASS', 'NV', 'KHONG_NV', '%PV/CV', '%Pass/PV', '%NV/Pass', '%NV/CV', '%Không đến PV', '%Tỷ trọng CV', '%Tỷ trọng NV']; dotc = ['CV_DOT', 'DOT_PV', 'DOT_PASS', 'DOT_NV', '%Đợt CV→PV', '%Đợt CV→Pass', '%Đợt CV→NV']
    for nm, key in (('3. Theo nguồn CV', 'by_source'), ('4. Theo nhân sự tuyển dụng (HR)', 'by_hr'), ('5. Theo chi nhánh / phòng ban (mọi vị trí)', 'by_unit')):
        S.append((nm + ' – theo kỳ', R[key][kyc])); S.append((nm + ' – theo đợt CV', R[key][dotc]))
    S.append(('5b. Theo khối', R['by_khoi'][['CV', 'PV', 'PASS', 'NV', '%PV/CV', '%Pass/PV', '%NV/Pass', '%NV/CV']]))
    S.append(('5c. Theo vị trí (top 15 theo CV)', R['by_pos'].sort_values('CV', ascending=False).head(15)[['CV', 'PV', 'PASS', 'NV', '%PV/CV', '%Pass/PV', '%NV/Pass', '%NV/CV']]))
    S.append(('6. Biến động nhân sự theo chi nhánh', bt))
    dr = R['dept_report']
    if len(dr):
        dd = dr.copy(); dd.loc['% Tham gia PV / CV'] = [pct(a, b) for a, b in zip(dr.loc['Tham gia PV'], dr.loc['Số CV'])]; dd.loc['% Đạt PV / Tham gia PV'] = [pct(a, b) for a, b in zip(dr.loc['Đạt PV'], dr.loc['Tham gia PV'])]; dd.loc['% Đã NV / Đạt PV'] = [pct(a, b) for a, b in zip(dr.loc['Đã NV'], dr.loc['Đạt PV'])]
        S.append(('7. Phòng ban / chi nhánh – đúng định nghĩa báo cáo', dd))
    S.append(('8. Người hết thử việc trong tháng (còn làm việc)', R['dat_thu_viec'].rename('Số người').to_frame()))
    rt = R['retention']
    S.append(('9. Giữ chân người nhận việc', pd.DataFrame({'Số lượng': [rt['n'], rt['nghi_all'], rt['nghi30'], rt['nghi60']], '%': [np.nan, pct(rt['nghi_all'], rt['n']), pct(rt['nghi30'], rt['n']), pct(rt['nghi60'], rt['n'])]}, index=pd.Index(['Nhận việc trong tháng', 'Đã nghỉ (tính đến nay)', 'Nghỉ trong 30 ngày', 'Nghỉ trong 60 ngày'], name='Chỉ tiêu'))))
    if len(R['retention_src']): S.append(('9b. Giữ chân theo nguồn', R['retention_src'].set_index('NGUON')[['NV', 'NGHI', 'NGHI30', '%Đã nghỉ', '%Nghỉ ≤30 ngày']]))
    n = R['nghi_total']
    S.append(('10. Nghỉ việc – tổng quan', pd.DataFrame({'Số lượng': [n, R['nghi_tv'], R['nghi_early10'], R['nghi_moi']], '%': [pct(n, T['NV']), pct(R['nghi_tv'], n), pct(R['nghi_early10'], n), pct(R['nghi_moi'], n)]},
                index=pd.Index(['Nghỉ việc (% so với nhận việc)', 'Nghỉ thử việc <30 ngày (% của nghỉ)', 'Nghỉ ≤10 ngày (% của nghỉ)', 'Vào và nghỉ trong cùng tháng (% của nghỉ)'], name='Chỉ tiêu'))))
    for nm, key in (('10b. Thời gian làm việc trước khi nghỉ', 'tenure'), ('10c. Nghỉ việc theo chi nhánh / bộ phận', 'nghi_unit'), ('10d. Nghỉ việc theo chức vụ (top 15)', 'nghi_pos'), ('10e. Nhóm lý do nghỉ', 'nghi_reason'), ('10f. Nghỉ việc theo nguồn', 'nghi_src')):
        if len(R[key]): S.append((nm, R[key].head(15) if key == 'nghi_pos' else R[key]))
    if len(R['nghi_excluded']): S.append(('10g. Người nghỉ trong nguồn nhưng báo cáo không liệt kê (cần xác minh)', R['nghi_excluded'][['TEN', 'CN', 'NNV', 'NNGHI', 'LY_DO_LOAI']].rename(columns={'TEN': 'Họ tên', 'CN': 'Chi nhánh', 'NNV': 'Ngày nhận việc', 'NNGHI': 'Ngày nghỉ', 'LY_DO_LOAI': 'Lý do không tính'}).reset_index(drop=True)))
    cost = {'Facebook Ads (VND)': c['fb'], 'Chi phí nhập tay (VND)': float(sum(c['manual'].values())), 'Tổng chi phí (VND)': c['total'], 'Chi phí / CV': c['per_cv'], 'Chi phí / lượt tham gia PV': c['per_pv'], 'Chi phí / Pass': c['per_pass'], 'Chi phí / người nhận việc': c['per_nv'],
            'Lượt nhắn tin Facebook': c['msg'], 'Chi phí / lượt nhắn tin': c['cpm'], 'Số ngày có số liệu Ads': c['ads_days'], 'Số ngày trong tháng': c['days_in_month']}
    for k_, v_ in c['manual'].items(): cost[f'  – {k_} (VND)'] = v_
    S.append(('11. Chi phí tuyển dụng', pd.DataFrame({'Giá trị': pd.Series(cost)})))
    if len(R['ads_daily']): S.append(('11b. Facebook Ads theo ngày', R['ads_daily'].rename(columns={'ngay': 'Ngày', 'spend': 'Chi tiêu (VND)', 'msg': 'Lượt nhắn tin'}).reset_index(drop=True)))
    cv_ = R['coverage']
    S.append(('12. Độ đầy đủ dữ liệu giữa các file', pd.DataFrame({'Số lượng': [cv_['nv_data'], cv_['nv_in_master'], cv_['nv_in_kt'], cv_['master_nv_month'], cv_['kt_nv_month'], cv_['nghi_n'], cv_['nghi_in_master'], cv_['nghi_in_ck'], cv_['nghi_in_kq'], cv_['nghi_lech_ngay'], cv_['nghi_lech_thang']]},
                index=pd.Index(['Người nhận việc (DATA)', '  – có trong Master', '  – có trong Kế toán (chi nhánh)', 'Dòng nhận việc tháng trong Master', 'Dòng nhận việc tháng trong Kế toán', 'Người nghỉ việc (hợp 3 nguồn)', '  – có trong Master', '  – có trong CHECK LIST', '  – có trong sheet NGHỈ VIỆC', 'Người nghỉ có ngày lệch giữa các file', '  – lệch sang tháng khác'], name='Chỉ tiêu'))))
    if Dp:
        S.append(('13. Sàng lọc CV theo kết quả đánh giá', Dp['screen'])); S.append(('13b. Sàng lọc CV theo nguồn', Dp['screen_src'])); S.append(('13c. Sàng lọc CV theo HR', Dp['screen_hr']))
        lt = Dp['lead']
        if lt['n']:
            S.append(('14. Thời gian tuyển (ngày)', lt['stats'])); S.append(('14b. Phân bố liên lạc → nhận việc', lt['dist'])); S.append(('14c. Thời gian tuyển theo nguồn', lt['by_source'])); S.append(('14d. Thời gian tuyển theo vị trí', lt['by_pos']))
        S.append(('15. Người phỏng vấn', Dp['interv']))
        for nm, key in (('16. Nguồn × chi nhánh – % nhận việc / CV', 'x_src_unit'), ('17. Nguồn × vị trí – % nhận việc / CV', 'x_src_pos')):
            if Dp[key] is not None: S.append((nm, Dp[key]['rate'])); S.append((nm.split(' – ')[0] + ' – số CV', Dp[key]['cv'])); S.append((nm.split(' – ')[0] + ' – số nhận việc', Dp[key]['nv']))
        sal = Dp['salary']
        if sal.get('n'): S.append((f"18. Lương đề nghị (có ghi lương: {sal['n']}/{sal['cv']} CV)", sal['table']))
        S.append(('19. Giữ chân – đường cong', Dp['ret']['curve']))
        for nm, key in (('19b. Nghỉ trong 30 ngày theo nguồn', 'by_source'), ('19c. theo chi nhánh / bộ phận', 'by_unit'), ('19d. theo vị trí', 'by_pos'), ('19e. theo HR', 'by_hr')): S.append((nm, Dp['ret'][key]))
    return [(t, d) for t, d in S if d is not None and len(d)]

def adj_lines(R):
    ad = R['adj']; L = []
    if not ad: return L
    L.append(f"Khối điều chỉnh: {ad.get('ten', '')}")
    for x in ad.get('nhan_viec_bo_sung', []): L.append(f"Thêm người nhận việc: {x['TEN']} ({x['CN']}, {x['NNV']}) – {x['ly_do']}")
    for k_, dt_, why in ad.get('loai_tru_nghi', []): L.append(f"Không tính vào nghỉ việc: {k_.title()} (nhận việc {dt_}) – {why}")
    ly = ad.get('ly_do_delta', {})
    for nm, key in (('Theo nguồn', 'delta_nguon'), ('Theo HR', 'delta_hr'), ('Theo chi nhánh (mọi vị trí)', 'delta_unit'), ('Bảng phòng ban/chi nhánh', 'delta_dept')):
        if ad.get(key): L.append(f"{nm}: " + '; '.join(f'{g_} – {mm}: {v:+d}' for (g_, mm), v in ad[key].items()) + f" | {ly.get(key.split('_')[1], '')}")
    L.append('Nhập tay theo báo cáo: đầu kỳ, số lượng yêu cầu, chi phí CareerViet/TopCV, Facebook Ads 30–31/10, nhóm lý do nghỉ.')
    for k_, (a_, b_) in R.get('adj_baseline_diff', {}).items(): L.append(f'CẢNH BÁO: raw hiện tại khác raw lúc lập điều chỉnh ở {k_}: {a_} (hiện tại) so với {b_} (khi lập).')
    return L

def _write_issues(ws, df, F, row=2):
    cols = list(df.columns); W = {'STT': 6, 'Tháng': 8, 'Mức độ': 11, 'Loại': 40, 'File': 30, 'Sheet': 30, 'Dòng': 8, 'Họ tên': 28, 'Chi tiết': 95, 'Gợi ý xử lý': 60}
    for j, c in enumerate(cols): ws.write(row, j, c, F['hdr']); ws.set_column(j, j, W.get(c, 16))
    fm = {'Cao': F['sev_hi'], 'Trung bình': F['sev_mid'], 'Thấp': F['sev_lo']}
    for i, rec in enumerate(df.itertuples(index=False), start=row + 1):
        for j, v in enumerate(rec):
            if v is None or (isinstance(v, float) and np.isnan(v)): ws.write_blank(i, j, None, F['txt'])
            elif cols[j] == 'Mức độ': ws.write_string(i, j, str(v), fm.get(v, F['txt']))
            elif isinstance(v, (int, float, np.integer, np.floating)): ws.write_number(i, j, float(v), F['int'])
            else: ws.write_string(i, j, str(v), F['wrap2'])
    ws.freeze_panes(row + 1, 2)
    if len(df): ws.autofilter(row, 0, row + len(df), len(cols) - 1)

def _write_weekly(ws, Wk, Kt, F, title):
    ws.set_column(0, 0, 34); ws.set_column(1, 24, 13); ws.write(0, 0, title, F['title']); r = 2
    wt = Wk['table'][['Từ', 'Đến', 'CV', 'PV', 'PASS', 'KHONG_DEN_PV', 'NV', 'KHONG_NV', 'NGHI', 'NGHI_TV', 'NGHI_SOM', '% PV/CV', '% Pass/PV', '% NV/Pass', '% NV/CV', '% Nghỉ/Nhận việc', '% Nghỉ thử việc/Nghỉ']]
    secs = [('1. Tổng hợp theo tuần (tuần * = bị cắt theo ranh giới tháng; dòng CẢ THÁNG = tổng các tuần)', wt)]
    for k_, nm_ in (('CV', 'Số CV'), ('PV', 'Tham gia PV'), ('PASS', 'Đạt PV'), ('NV', 'Nhận việc')): secs.append((f'2. Theo HR × tuần – {nm_}', Wk['hr'][k_]))
    for k_, nm_ in (('CV', 'Số CV'), ('PASS', 'Đạt PV'), ('NV', 'Nhận việc')): secs.append((f'3. Theo nguồn × tuần – {nm_}', Wk['src'][k_]))
    for k_, nm_ in (('CV', 'Số CV'), ('NV', 'Nhận việc'), ('NGHI', 'Nghỉ việc')): secs.append((f'4. Theo chi nhánh / bộ phận × tuần – {nm_}', Wk['unit'][k_]))
    if Kt is not None:
        secs.append(('5. KTV / TVV theo chi nhánh (cả tháng): Đã nhận việc và nghỉ', Kt['cn']))
        if len(Kt['tenure']): secs.append(('6. Thời gian làm việc trước khi nghỉ (KTV / TVV)', Kt['tenure']))
    for ttl, df in secs:
        if df is None or not len(df): continue
        ws.merge_range(r, 0, r, 8, ttl, F['sec']); r = _write_df(ws, df, r + 1, F)
    ws.freeze_panes(3, 1)

def export_excel(results, deeps=None, bts=None, trend=None, meta=None, weekly=None, checks=None, custom=None):
    """Xuất toàn bộ báo cáo các tháng đã chọn ra Excel (bytes): mỗi tháng một sheet báo cáo + 3 sheet danh sách; thêm So sánh, Xu hướng, Điều chỉnh, Ghi chú."""
    deeps = deeps or {}; bts = bts or {}; meta = meta or {}; buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='xlsxwriter', datetime_format='dd/mm/yyyy') as w:
        wb = w.book
        F = dict(title=wb.add_format({'bold': True, 'font_size': 14, 'font_color': '#1F3864'}), sec=wb.add_format({'bold': True, 'font_color': '#FFFFFF', 'bg_color': '#2F5597'}),
                 hdr=wb.add_format({'bold': True, 'bg_color': '#D9E1F2', 'border': 1, 'text_wrap': True, 'valign': 'vcenter', 'align': 'center'}), txt=wb.add_format({'border': 1}),
                 int=wb.add_format({'border': 1, 'num_format': '#,##0'}), flt=wb.add_format({'border': 1, 'num_format': '#,##0.0'}), pct=wb.add_format({'border': 1, 'num_format': '0.0"%"'}),
                 date=wb.add_format({'border': 1, 'num_format': 'dd/mm/yyyy'}), sev_hi=wb.add_format({'border': 1, 'bg_color': '#FFC7CE'}), sev_mid=wb.add_format({'border': 1, 'bg_color': '#FFEB9C'}), sev_lo=wb.add_format({'border': 1, 'bg_color': '#EDEDED'}), wrap2=wb.add_format({'border': 1, 'text_wrap': True, 'valign': 'top'}), note=wb.add_format({'italic': True, 'font_color': '#595959', 'text_wrap': True, 'valign': 'top'}), wrap=wb.add_format({'text_wrap': True, 'valign': 'top'}))
        def sheet(name):
            ws = wb.add_worksheet(name[:31]); w.sheets[name[:31]] = ws; return ws
        # --- Tổng quan
        ws = sheet('Tổng quan'); ws.set_column(0, 0, 36); ws.set_column(1, 1, 90); ws.write(0, 0, 'BÁO CÁO TUYỂN DỤNG THEO THÁNG', F['title'])
        info = [('Ngày tạo file', pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')), ('Năm', str(results[0]['year']) if results else ''), ('Các tháng', ', '.join(f"T{R['month']:02d}" for R in results)),
                ('Ngày dữ liệu mới nhất', f"{results[0]['cutoff']:%d/%m/%Y}" if results else ''), ('Khớp báo cáo đã duyệt', 'Có (tháng có khối điều chỉnh)' if any(R['adjusted'] for R in results) else 'Không (số thuần từ raw)')]
        for k_, v_ in meta.get('files', {}).items(): info.append((f'File nguồn – {k_}', str(v_)))
        for R in results:
            if R['adjusted']: info.append((f"T{R['month']:02d} đã áp dụng điều chỉnh", R['adj'].get('ten', '')))
        for i, (k_, v_) in enumerate(info, start=2): ws.write(i, 0, k_, F['hdr']); ws.write(i, 1, v_, F['txt'])
        r0 = len(info) + 3; ws.write(r0, 0, 'Nội dung các sheet', F['sec']); ws.write(r0, 1, '', F['sec'])
        for i, (a_, b_) in enumerate([('So sánh tháng', 'Chỉ tiêu chính của các tháng và % thay đổi'), ('Xu hướng năm', '12 tháng của năm (tính từ raw)'), ('Txx Báo cáo', 'Toàn bộ bảng của tháng xx, đánh số mục 1–19'), ('Txx DS nhận việc / DS nghỉ việc / DS đạt TV', 'Danh sách từng người'), ('Txx Theo tuần', 'Số liệu từng tuần (thứ Hai–Chủ nhật), theo HR/nguồn/chi nhánh, KTV–TVV'), ('Txx Kiểm tra', 'Tên nghi nhập sai, người chưa nhập, ngày/chi nhánh lệch giữa DATA–Kế toán–Master'), ('Kiểm tra (tất cả)', 'Gộp các vấn đề của mọi tháng đã chọn'), ('Khoảng ngày', 'Báo cáo theo khoảng ngày tùy chọn (nếu bật)'), ('Điều chỉnh', 'Các điều chỉnh để khớp báo cáo và lý do'), ('Ghi chú', 'Định nghĩa và lưu ý')], start=r0 + 1): ws.write(i, 0, a_, F['txt']); ws.write(i, 1, b_, F['txt'])
        # --- So sánh
        ws = sheet('So sánh tháng'); ws.set_column(0, 0, 38); ws.set_column(1, 12, 16); ws.write(0, 0, 'SO SÁNH CÁC THÁNG', F['title']); _write_df(ws, compare(results), 2, F); ws.freeze_panes(3, 1)
        # --- Xu hướng
        if trend is not None:
            ws = sheet('Xu hướng năm'); ws.set_column(0, 0, 30); ws.set_column(1, 20, 14); ws.write(0, 0, f"XU HƯỚNG 12 THÁNG NĂM {results[0]['year']} (từ raw, không gồm điều chỉnh khớp báo cáo)", F['title']); r = 2
            for ttl, df in (('Số lượng và tỷ lệ theo tháng', trend['table']), ('Thay đổi so với tháng trước (%)', trend['mom']), ('Nhận việc theo nguồn (6 nguồn nhiều nhất)', trend['src_nv']), ('CV theo nguồn (8 nguồn nhiều nhất)', trend['src_cv'])):
                ws.merge_range(r, 0, r, 8, ttl, F['sec']); r = _write_df(ws, df, r + 1, F)
        # --- Từng tháng
        for R in results:
            m = R['month']; p = f'T{m:02d}'; Dp = deeps.get(m); bt = bts.get(m)
            if bt is None: bt = branch_table(R, (R['adj'] or {}).get('dau_ky'), (R['adj'] or {}).get('sl_yc'))
            ws = sheet(f'{p} Báo cáo'); ws.set_column(0, 0, 42); ws.set_column(1, 14, 15); widths = {}
            ws.write(0, 0, f"BÁO CÁO TUYỂN DỤNG THÁNG {m:02d}/{R['year']} ({R['start']:%d/%m} – {R['end']:%d/%m})", F['title'])
            ws.write(1, 0, 'Khớp báo cáo đã duyệt: ' + ('có (xem sheet Điều chỉnh)' if R['adjusted'] else 'không – số thuần từ raw'), F['note']); r = 3
            if not R['has_data']: ws.write(r, 0, 'Không có CV hoặc nhận việc nào trong tháng này (file chưa cập nhật tới tháng này?).', F['note']); r += 2
            for ttl, df in month_sections(R, Dp, bt):
                ws.merge_range(r, 0, r, 8, ttl, F['sec']); r = _write_df(ws, df, r + 1, F)
            ws.freeze_panes(3, 1)
            # danh sách
            nv = R['nv_list']; ws = sheet(f'{p} DS nhận việc'); wd = {}
            ws.write(0, 0, f'DANH SÁCH NGƯỜI NHẬN VIỆC THÁNG {m:02d}/{R["year"]}', F['title'])
            cols = {'TEN': 'Họ tên', 'KHOI': 'Khối', 'CN': 'Chi nhánh / bộ phận', 'VITRI': 'Vị trí', 'NGUON': 'Nguồn', 'LH': 'Người liên hệ', 'NNV': 'Ngày nhận việc', 'NGAY_DEN_KHI_NGHI': 'Số ngày đến khi nghỉ', 'DA_NGHI': 'Đã nghỉ', 'GHI_CHU': 'Ghi chú'}
            _write_df(ws, nv[list(cols)].rename(columns=cols).reset_index(drop=True) if len(nv) else pd.DataFrame(columns=list(cols.values())), 2, F, widths=wd)
            for j, wv in wd.items(): ws.set_column(j, j, max(12, wv + 2))
            ws.freeze_panes(3, 1); ng = R['nghi_list']; ws = sheet(f'{p} DS nghỉ việc'); wd = {}
            ws.write(0, 0, f'DANH SÁCH NGƯỜI NGHỈ VIỆC THÁNG {m:02d}/{R["year"]}', F['title'])
            cols = {'TEN': 'Họ tên', 'CN': 'Chi nhánh', 'TT': 'Tình trạng', 'MSNV': 'MSNV', 'BP': 'Bộ phận', 'CV': 'Chức vụ', 'NNV': 'Ngày nhận việc', 'NNGHI': 'Ngày nghỉ', 'days': 'Số ngày làm việc', 'NHOM': 'Nhóm thâm niên', 'LOAI': 'Loại', 'NGUON': 'Nguồn', 'LH': 'Người liên hệ', 'NHOM_LY_DO': 'Nhóm lý do (tự động)', 'LYDO': 'Lý do', 'src': 'Có trong file', 'LECH_NGAY': 'Ngày nghỉ lệch giữa các file'}
            _write_df(ws, ng[list(cols)].rename(columns=cols).reset_index(drop=True) if len(ng) else pd.DataFrame(columns=list(cols.values())), 2, F, widths=wd)
            for j, wv in wd.items(): ws.set_column(j, j, max(12, wv + 2))
            ws.freeze_panes(3, 1); dl = R['dat_thu_viec_list']; ws = sheet(f'{p} DS đạt TV'); wd = {}
            ws.write(0, 0, f'DANH SÁCH NGƯỜI HẾT THỬ VIỆC THÁNG {m:02d}/{R["year"]}', F['title'])
            cols = {'CN_BC': 'Chi nhánh', 'TT': 'Tình trạng', 'MSNV': 'MSNV', 'TEN': 'Họ tên', 'BP': 'Bộ phận', 'CV': 'Chức vụ', 'NNV_d': 'Ngày nhận việc', 'NCTV_d': 'Ngày cuối thử việc'}
            _write_df(ws, dl[list(cols)].rename(columns=cols).reset_index(drop=True) if len(dl) else pd.DataFrame(columns=list(cols.values())), 2, F, widths=wd)
            for j, wv in wd.items(): ws.set_column(j, j, max(12, wv + 2))
            ws.freeze_panes(3, 1)
        for R in results:
            m = R['month']; p = f'T{m:02d}'
            if weekly and m in weekly:
                ws = sheet(f'{p} Theo tuần'); _write_weekly(ws, weekly[m]['week'], weekly[m].get('ktv'), F, f"SỐ LIỆU THEO TUẦN – THÁNG {m:02d}/{R['year']} (từ raw, không gồm điều chỉnh khớp báo cáo)")
            if checks and m in checks:
                ws = sheet(f'{p} Kiểm tra'); ws.write(0, 0, f"KIỂM TRA DỮ LIỆU THÁNG {m:02d}/{R['year']} – tên nghi nhập sai, người chưa nhập, ngày/chi nhánh lệch", F['title']); _write_issues(ws, checks[m], F)
        if checks:
            allc = []
            for R in results:
                if R['month'] in checks: c_ = checks[R['month']].copy(); c_.insert(1, 'Tháng', f"T{R['month']:02d}"); allc.append(c_)
            if allc:
                ws = sheet('Kiểm tra (tất cả)'); ws.write(0, 0, 'KIỂM TRA DỮ LIỆU – TẤT CẢ CÁC THÁNG ĐÃ CHỌN', F['title']); _write_issues(ws, pd.concat(allc, ignore_index=True), F)
        if custom:
            Rr, Dr = custom; ws = sheet('Khoảng ngày'); ws.set_column(0, 0, 42); ws.set_column(1, 14, 15); ws.write(0, 0, f"BÁO CÁO THEO KHOẢNG NGÀY: {Rr['label']}", F['title']); ws.write(1, 0, 'Số thuần từ raw (không gồm điều chỉnh khớp báo cáo).', F['note']); r = 3
            for ttl, df in month_sections(Rr, Dr, branch_table(Rr, None, None)): ws.merge_range(r, 0, r, 8, ttl, F['sec']); r = _write_df(ws, df, r + 1, F)
            ws.freeze_panes(3, 1)
        # --- Điều chỉnh + Ghi chú
        ws = sheet('Điều chỉnh'); ws.set_column(0, 0, 14); ws.set_column(1, 1, 150); ws.write(0, 0, 'CÁC ĐIỀU CHỈNH ĐỂ KHỚP BÁO CÁO ĐÃ DUYỆT', F['title']); r = 2
        any_adj = False
        for R in results:
            if R['adjusted']:
                any_adj = True
                for line in adj_lines(R): ws.write(r, 0, f"T{R['month']:02d}", F['txt']); ws.write(r, 1, line, F['wrap']); r += 1
        if not any_adj: ws.write(r, 0, 'Không có điều chỉnh nào được áp dụng – toàn bộ là số thuần từ raw.', F['note'])
        ws = sheet('Ghi chú'); ws.set_column(0, 0, 160); ws.write(0, 0, 'ĐỊNH NGHĨA VÀ LƯU Ý', F['title'])
        for i, t_ in enumerate(XL_NOTES, start=2): ws.write(i, 0, t_, F['wrap'])
    return buf.getvalue()




# ------------------------------------------------------------------ THEO TUẦN / KTV-TVV / KIỂM TRA DỮ LIỆU
def iso_weeks(y, m):
    """Các tuần (thứ Hai → Chủ nhật, số tuần theo năm như sheet Kế toán 'TUẦN 40') cắt theo ranh giới tháng."""
    s, e = month_range(y, m); out = []; cur = s
    while cur <= e:
        end = min(cur + pd.Timedelta(days=6 - cur.weekday()), e); iso = cur.isocalendar()
        full = (cur.weekday() == 0) and (end.weekday() == 6)
        out.append(dict(no=int(iso.week), start=cur, end=end, partial=not full, label=f'Tuần {int(iso.week)} ({cur:%d/%m}–{end:%d/%m})' + ('' if full else ' *')))
        cur = end + pd.Timedelta(days=1)
    return out

def _funnel_counts(d, s, e):
    pvm = _in(d.NGAYPV_d, s, e); nvm = _in(d.NGAYNV_d, s, e)
    return dict(CV=int(_in(d.NGAYLL_d, s, e).sum()), PV=int((pvm & d.KQPV.isin(PVOK)).sum()), PASS=int((pvm & (d.KQPV == 'Đạt')).sum()), KHONG_DEN_PV=int((pvm & (d.KQPV == 'Không đến pv')).sum()),
                NV=int((nvm & (d.KQNV == 'Đã nhận việc')).sum()), KHONG_NV=int((nvm & (d.KQNV == 'Không nhận việc')).sum()))

def weekly_summary(raw, y, m):
    """Số liệu theo tuần của một tháng: tổng, theo HR, theo nguồn, theo chi nhánh. Tổng các tuần = số của tháng."""
    d = raw['d']; nga = raw['nghi_all']; W = iso_weeks(y, m); rows = []; hr = {k: {} for k in ('CV', 'PV', 'PASS', 'NV')}; src = {k: {} for k in ('CV', 'PASS', 'NV')}; unit = {k: {} for k in ('CV', 'NV', 'NGHI')}
    for w in W:
        s, e = w['start'], w['end']; T = _funnel_counts(d, s, e); ng = nga[_in(nga.NNGHI, s, e)] if len(nga) else nga
        T.update(NGHI=len(ng), NGHI_TV=int((ng.days < 30).sum()) if len(ng) else 0, NGHI_SOM=int((ng.days <= 10).sum()) if len(ng) else 0, Tuần=w['label'], Từ=s, Đến=e); rows.append(T)
        pvm = _in(d.NGAYPV_d, s, e); nvm = _in(d.NGAYNV_d, s, e) & (d.KQNV == 'Đã nhận việc')
        sets = {'CV': d[_in(d.NGAYLL_d, s, e)], 'PV': d[pvm & d.KQPV.isin(PVOK)], 'PASS': d[pvm & (d.KQPV == 'Đạt')], 'NV': d[nvm]}
        for k in hr: hr[k][w['label']] = sets[k].groupby('LH').size()
        for k in src: src[k][w['label']] = sets[k].groupby('NGUON').size()
        for k in ('CV', 'NV'): unit[k][w['label']] = sets[k].groupby('DON_VI').size()
        unit['NGHI'][w['label']] = ng.groupby('CN').size() if len(ng) else pd.Series(dtype=int)
    t = pd.DataFrame(rows).set_index('Tuần'); cnt = ['CV', 'PV', 'PASS', 'KHONG_DEN_PV', 'NV', 'KHONG_NV', 'NGHI', 'NGHI_TV', 'NGHI_SOM']
    tot = t[cnt].sum(); t.loc['CẢ THÁNG'] = list(tot[cnt]) + [pd.NaT, pd.NaT]; t[cnt] = t[cnt].astype(int)
    t['% PV/CV'] = [pct(a, b) for a, b in zip(t.PV, t.CV)]; t['% Pass/PV'] = [pct(a, b) for a, b in zip(t.PASS, t.PV)]; t['% NV/Pass'] = [pct(a, b) for a, b in zip(t.NV, t.PASS)]; t['% NV/CV'] = [pct(a, b) for a, b in zip(t.NV, t.CV)]
    t['% Nghỉ/Nhận việc'] = [pct(a, b) for a, b in zip(t.NGHI, t.NV)]; t['% Nghỉ thử việc/Nghỉ'] = [pct(a, b) for a, b in zip(t.NGHI_TV, t.NGHI)]
    def piv(dct):
        out = {}
        for k, cols in dct.items():
            df = pd.DataFrame(cols).fillna(0).astype(int); df['Cả tháng'] = df.sum(axis=1); out[k] = df.sort_values('Cả tháng', ascending=False)
        return out
    return dict(weeks=W, table=t, hr=piv(hr), src=piv(src), unit=piv(unit))

def nhom_vi_tri(t):
    """Gom vị trí/chức vụ thành KTV / TVV / Khác (gõ sai dấu vẫn nhận: 'Tư vấn viến' → TVV)."""
    n = norm(t)
    if re.match(r'^(KTV|KY THUAT VIE)', n): return 'KTV'
    if re.match(r'^(TVV|TU VAN VIE)', n): return 'TVV'
    return 'Khác'
TENURE_ALT = ['Từ 1–3 ngày', 'Từ 4–7 ngày', 'Từ 8–30 ngày', 'Từ 31–90 ngày', 'Trên 90 ngày']
def bucket_alt(n): return TENURE_ALT[0] if n <= 3 else TENURE_ALT[1] if n <= 7 else TENURE_ALT[2] if n <= 30 else TENURE_ALT[3] if n <= 90 else TENURE_ALT[4]
def ktv_tvv(R):
    """Bảng kiểu dashboard theo dõi: theo chi nhánh × KTV/TVV: Đã nhận việc, nghỉ thử việc / chính thức; kèm thời gian làm việc trước khi nghỉ (1–3, 4–7, 8–30, 31–90, >90 ngày)."""
    nv = R['nv_list'].copy(); ng = R['nghi_list'].copy(); out = {}
    if len(nv): nv['NHOM'] = nv.VITRI.map(nhom_vi_tri); nv['CN_N'] = nv.CN.map(norm)
    if len(ng): ng['NHOM'] = ng.CV.map(nhom_vi_tri); ng['CN_N'] = ng.CN.map(norm); ng['TV'] = ng.days < 30
    rows = []
    for b in BRANCHES + ['VĂN PHÒNG / KHÁC']:
        key = norm(b); rec = {'Chi nhánh': b}
        for g in ('KTV', 'TVV', 'Khác'):
            n_ = nv[(nv.NHOM == g) & ((nv.CN_N == key) if b != 'VĂN PHÒNG / KHÁC' else ~nv.CN_N.isin([norm(x) for x in BRANCHES]))] if len(nv) else nv
            g_ = ng[(ng.NHOM == g) & ((ng.CN_N == key) if b != 'VĂN PHÒNG / KHÁC' else ~ng.CN_N.isin([norm(x) for x in BRANCHES]))] if len(ng) else ng
            rec[f'{g} – Đã NV'] = len(n_); rec[f'{g} – Nghỉ thử việc'] = int(g_.TV.sum()) if len(g_) else 0; rec[f'{g} – Nghỉ chính thức'] = int((~g_.TV).sum()) if len(g_) else 0; rec[f'{g} – Nghỉ tổng'] = len(g_)
        rows.append(rec)
    t = pd.DataFrame(rows).set_index('Chi nhánh'); t.loc['TỔNG'] = t.sum(); out['cn'] = t
    if len(ng):
        ng['NHOM_ALT'] = ng.days.map(lambda x: bucket_alt(x) if pd.notna(x) else None)
        tt = pd.DataFrame({g: ng[ng.NHOM == g].NHOM_ALT.value_counts().reindex(TENURE_ALT).fillna(0).astype(int) for g in ('KTV', 'TVV', 'Khác')}); tt['Tổng'] = tt.sum(axis=1)
        for g in list(tt.columns): tt[f'% {g}'] = [pct(a, tt[g].sum()) for a in tt[g]]
        tt.index.name = 'Thời gian làm việc'; out['tenure'] = tt
    else: out['tenure'] = pd.DataFrame()
    out['khac_nv'] = nv[nv.NHOM == 'Khác'].VITRI.value_counts() if len(nv) else pd.Series(dtype=int); out['khac_ng'] = ng[ng.NHOM == 'Khác'].CV.value_counts() if len(ng) else pd.Series(dtype=int)
    return out

SIM_MIN = 0.80
def _sim(a, b): return difflib.SequenceMatcher(None, a, b).ratio()
SEV_ORDER = {'Cao': 0, 'Trung bình': 1, 'Thấp': 2}
def kiem_tra_du_lieu(raw, s, e):
    """Soát dữ liệu giữa DATA / Kế toán / Master cho kỳ [s, e]: tên nghi nhập sai, người chưa nhập, ngày/chi nhánh lệch, ngày vô lý. Mỗi lỗi kèm file, sheet, số dòng và gợi ý sửa."""
    d = raw['d']; ms = raw['ms']; kn = raw['kn']; kq = raw['kq']; ck = raw['ck']; nga = raw['nghi_all']; cut = raw['cutoff']; I = []
    def add(sev, typ, file, sheet, row, name, detail, hint): I.append({'Mức độ': sev, 'Loại': typ, 'File': file, 'Sheet': sheet, 'Dòng': (int(row) if row is not None and not (isinstance(row, float) and np.isnan(row)) else None), 'Họ tên': name, 'Chi tiết': detail, 'Gợi ý xử lý': hint})
    fd = lambda x: '' if pd.isna(x) else pd.Timestamp(x).strftime('%d/%m/%Y')
    def near(df, col, dt, days): return df[(df[col] - dt).abs() <= pd.Timedelta(days=days)] if pd.notna(dt) else df.iloc[0:0]
    # ---- 1) người nhận việc (DATA) có trong Master / Kế toán không; sai tên?
    nv = d[_in(d.NGAYNV_d, s, e) & (d.KQNV == 'Đã nhận việc')].drop_duplicates(['k', 'NGAYNV_d'])
    for r in nv.itertuples():
        for nm, df, skip_vp in (('Master Data', ms, False), ('Kế toán Admin (sheet NHẬN VIỆC)', kn, True)):
            if skip_vp and r.KHOI == 'Văn Phòng': continue
            win = near(df, 'NNV_d', r.NGAYNV_d, 7)
            if len(win[win.k == r.k]): continue
            cand = win.assign(SIMV=[_sim(r.k, x) for x in win.k])
            same_cn = lambda a, b: norm(a) == norm(b) or {norm(a), norm(b)} == {'BINH DUONG', 'BINH DUOG'}
            cand = cand[np.array([((cc.NNV_d == r.NGAYNV_d) and cc.SIMV >= 0.85) or (same_cn(cc.CN, r.CN) and cc.SIMV >= 0.90 and abs((cc.NNV_d - r.NGAYNV_d).days) <= 3) for cc in cand.itertuples()], dtype=bool)].sort_values('SIMV', ascending=False)
            if len(cand):
                c = cand.iloc[0]; other = kn if df is ms else ms; oth = near(other, 'NNV_d', r.NGAYNV_d, 7); in_other_same = bool(len(oth[oth.k == r.k])); in_other_cand = bool(len(oth[oth.k == c.k]))
                who = 'File kia (Kế toán/Master) cũng ghi giống DATA → file "' + nm + '" nhiều khả năng gõ sai' if in_other_same else ('File kia ghi giống ' + nm + ' → DATA nhiều khả năng gõ sai' if in_other_cand else 'Chưa đủ căn cứ biết bên nào đúng')
                add('Cao', 'Tên nghi nhập sai (nhận việc)', nm, 'DATA TUYỂN DỤNG ↔ ' + nm, (c.ROW if 'ROW' in c.index else (c.name + 2)), r.TEN, f'DATA dòng {r.Index + 2} ghi "{r.TEN}" (nhận việc {fd(r.NGAYNV_d)}); {nm} dòng {(c.ROW if "ROW" in c.index else c.name + 2)} ghi "{c.TEN}" (giống {c.SIMV:.0%}, ngày nhận việc {fd(c.NNV_d)}, chi nhánh {c.CN})', who + '. Sửa cho cùng một cách viết.')
            else: add('Trung bình', 'Chưa nhập vào file khác (nhận việc)', nm, 'DATA TUYỂN DỤNG → ' + nm, r.Index + 2, r.TEN, f'Có "Đã nhận việc" ngày {fd(r.NGAYNV_d)} ở DATA (chi nhánh {r.CN}, vị trí {r.VITRI}) nhưng không thấy trong {nm}.', f'Bổ sung người này vào {nm}.')
    # ---- 2) người nghỉ: thiếu ở file nào, bị tách đôi do sai tên, ngày vô lý
    ng = nga[_in(nga.NNGHI, s, e)].copy() if len(nga) else nga
    for r in ng.itertuples():
        miss = [x for x, f_ in (('Master', 'Master'), ('CHECK LIST', 'CHECK LIST'), ('NGHỈ VIỆC (Kế toán)', 'KT NGHỈ VIỆC')) if f_ not in r.src and not (f_ == 'KT NGHỈ VIỆC' and norm(r.CN) == 'VAN PHONG')]
        if miss: add('Trung bình', 'Người nghỉ chưa nhập đủ ở các file', ', '.join(miss), 'Nghỉ việc', None, r.TEN, f'Nghỉ ngày {fd(r.NNGHI)} (chi nhánh {r.CN}); chỉ có trong: {r.src}.', 'Bổ sung ngày nghỉ vào file còn thiếu: ' + ', '.join(miss))
        if pd.notna(r.NNV) and r.NNGHI < r.NNV: add('Cao', 'Ngày nghỉ trước ngày nhận việc', r.src, 'Nghỉ việc', None, r.TEN, f'Nhận việc {fd(r.NNV)} nhưng nghỉ {fd(r.NNGHI)} (sớm hơn {int((r.NNV - r.NNGHI).days)} ngày).', 'Kiểm tra lại ngày nhận việc hoặc ngày nghỉ (thường gõ nhầm tháng/năm).')
        if getattr(r, 'LECH_NGAY', False): add('Trung bình', 'Ngày nghỉ khác nhau giữa các file', r.src, 'Nghỉ việc', None, r.TEN, f'Các file ghi ngày nghỉ khác nhau (app lấy {fd(r.NNGHI)}, ưu tiên Master → CHECK LIST → NGHỈ VIỆC).', 'Thống nhất một ngày nghỉ trong cả 3 file.')
    ngs = ng.sort_values('NNGHI').reset_index(drop=True)
    for i in range(len(ngs)):
        for j in range(i + 1, len(ngs)):
            if (ngs.NNGHI[j] - ngs.NNGHI[i]).days > 3: break
            a, b = ngs.iloc[i], ngs.iloc[j]
            if a.k != b.k and _sim(a.k, b.k) >= 0.85 and (pd.isna(a.NNV) or pd.isna(b.NNV) or abs((a.NNV - b.NNV).days) <= 7):
                add('Cao', 'Một người nghỉ bị tách thành 2 vì sai tên (đang đếm 2 lần)', f'{a.src} | {b.src}', 'Nghỉ việc', None, f'{a.TEN} / {b.TEN}', f'"{a.TEN}" ({a.src}, nghỉ {fd(a.NNGHI)}) và "{b.TEN}" ({b.src}, nghỉ {fd(b.NNGHI)}) giống {_sim(a.k, b.k):.0%}, cùng ngày nhận việc gần nhau → nhiều khả năng là một người.', 'Sửa tên cho giống nhau; số nghỉ việc sẽ giảm 1 khi gộp.')
    # ---- 3) logic ngày trong DATA
    x = d[_in(d.NGAYLL_d, s, e) | _in(d.NGAYPV_d, s, e) | _in(d.NGAYNV_d, s, e)]
    def rows(mask, sev, typ, detail_fn, hint):
        for r in x[mask.reindex(x.index, fill_value=False)].itertuples(): add(sev, typ, 'TUYỂN DỤNG - DATA', 'DATA TUYỂN DỤNG', r.Index + 2, r.TEN, detail_fn(r), hint)
    rows(x.NGAYPV_d < x.NGAYLL_d, 'Trung bình', 'Ngày PV sớm hơn ngày liên lạc', lambda r: f'Liên lạc {fd(r.NGAYLL_d)} nhưng PV {fd(r.NGAYPV_d)}.', 'Thường gõ nhầm ngày/tháng; sửa lại một trong hai ngày.')
    rows(x.NGAYNV_d < x.NGAYPV_d, 'Trung bình', 'Ngày nhận việc sớm hơn ngày PV', lambda r: f'PV {fd(r.NGAYPV_d)} nhưng nhận việc {fd(r.NGAYNV_d)}.', 'Sửa lại ngày PV hoặc ngày nhận việc.')
    rows((x.KQNV == 'Đã nhận việc') & x.NGAYNV_d.isna(), 'Cao', 'Đã nhận việc nhưng thiếu ngày nhận việc', lambda r: f'Kết quả "Đã nhận việc", liên lạc {fd(r.NGAYLL_d)}, chi nhánh {r.CN}; không có ngày nhận việc nên không vào số liệu tuần/tháng nào.', 'Điền ngày nhận việc.')
    rows(x.KQPV.isin(PVOK) & x.NGAYPV_d.isna(), 'Trung bình', 'Có kết quả PV nhưng thiếu ngày PV', lambda r: f'Kết quả PV "{r.KQPV}", liên lạc {fd(r.NGAYLL_d)}; không có ngày PV nên không được tính vào Tham gia PV / Pass.', 'Điền ngày PV.')
    rows(x.NGAYNV_d.notna() & x.KQNV.isna(), 'Trung bình', 'Có ngày nhận việc nhưng chưa có kết quả nhận việc', lambda r: f'Ngày nhận việc {fd(r.NGAYNV_d)} nhưng ô kết quả trống.', 'Chọn "Đã nhận việc" hoặc "Không nhận việc".')
    rows(_in(x.NGAYLL_d, s, e) & (x.NGUON == '(không ghi nguồn)'), 'Thấp', 'Thiếu nguồn CV', lambda r: f'Liên lạc {fd(r.NGAYLL_d)}, chi nhánh {r.CN}.', 'Điền nguồn để tính đúng hiệu quả từng nguồn.')
    rows(_in(x.NGAYLL_d, s, e) & (x.LH == '(không ghi người LH)'), 'Thấp', 'Thiếu người liên hệ', lambda r: f'Liên lạc {fd(r.NGAYLL_d)}, nguồn {r.NGUON}.', 'Điền người liên hệ (HR).')
    for col, nm in (('NGAYLL_d', 'liên lạc'), ('NGAYPV_d', 'PV'), ('NGAYNV_d', 'nhận việc')):
        bad = x[(x[col] > cut + pd.Timedelta(days=35)) | (x[col] < pd.Timestamp('2020-01-01'))]
        for r in bad.itertuples(): add('Cao', 'Ngày bất thường (sai năm?)', 'TUYỂN DỤNG - DATA', 'DATA TUYỂN DỤNG', r.Index + 2, r.TEN, f'Ngày {nm} = {fd(getattr(r, col))}, ngoài khoảng hợp lý (dữ liệu mới nhất {fd(cut)}).', 'Kiểm tra lại năm/tháng gõ nhầm.')
    for col, nm in (('NGAYPV', 'NGÀY PV'), ('NGAYNV', 'NGÀY NHẬN VIỆC'), ('NGAYLL', 'NGÀY LIÊN LẠC')):
        t_ = x[x[col].map(lambda v: isinstance(v, str)).astype(bool) & x[col + '_d'].notna()]
        for r in t_.itertuples(): add('Thấp', 'Ô ngày nhập dạng chữ', 'TUYỂN DỤNG - DATA', 'DATA TUYỂN DỤNG', r.Index + 2, r.TEN, f'Cột {nm} là chữ "{getattr(r, col)}" (app đọc theo ngày/tháng/năm = {fd(getattr(r, col + "_d"))}); công thức Excel sẽ bỏ qua ô này.', 'Nhập lại thành ngày chuẩn (không gõ dạng chữ).')
    dup = d[_in(d.NGAYLL_d, s, e)]; dup = dup[dup.duplicated(['TEN', 'VITRI', 'NGAYLL_d', 'NGUON'], keep=False)].sort_values('TEN')
    for k_, g in dup.groupby(['TEN', 'VITRI', 'NGAYLL_d', 'NGUON']): add('Thấp', 'Dòng CV trùng', 'TUYỂN DỤNG - DATA', 'DATA TUYỂN DỤNG', g.index[0] + 2, k_[0], f'{len(g)} dòng giống nhau (vị trí {k_[1]}, liên lạc {fd(k_[2])}, nguồn {k_[3]}) tại các dòng {", ".join(str(i + 2) for i in g.index)}.', 'Xóa dòng thừa nếu là cùng một ứng viên (đang đếm vào số CV).')
    # ---- 4) lệch chi nhánh / ngày nhận việc giữa các file (người nhận việc trong kỳ)
    for r in nv.itertuples():
        for nm, df, skip_vp in (('Master Data', ms, False), ('Kế toán Admin (NHẬN VIỆC)', kn, True)):
            if skip_vp and r.KHOI == 'Văn Phòng': continue
            w_ = near(df[df.k == r.k], 'NNV_d', r.NGAYNV_d, 7)
            if not len(w_): continue
            c = w_.iloc[0]; row = c.ROW if 'ROW' in c.index else c.name + 2
            if c.NNV_d != r.NGAYNV_d: add('Trung bình', 'Ngày nhận việc lệch giữa các file', nm, nm, row, r.TEN, f'DATA {fd(r.NGAYNV_d)} ≠ {nm} {fd(c.NNV_d)}.', 'Thống nhất ngày nhận việc.')
            if r.KHOI != 'Văn Phòng' and norm(c.CN) not in (norm(r.CN),) and not ({norm(c.CN), norm(r.CN)} == {'BINH DUONG', 'BINH DUOG'}): add('Trung bình', 'Chi nhánh lệch giữa các file', nm, nm, row, r.TEN, f'DATA ghi {r.CN} ≠ {nm} ghi {c.CN}.', 'Xác nhận chi nhánh thật (hoặc ghi chú điều chuyển).')
            if nm == 'Kế toán Admin (NHẬN VIỆC)' and norm(c.CN) == 'BINH DUOG': add('Thấp', 'Sai chính tả tên chi nhánh', nm, nm, row, r.TEN, 'Chi nhánh ghi "Bình Dươg".', 'Sửa thành "Bình Dương".')
    # ---- 5) Master: trạng thái ↔ ngày nghỉ
    mt = ms[ms.TT.isin(['Đã nghỉ việc', 'Nghỉ ngang', 'Công ty cho nghỉ']) & ms.NNGHI_d.isna() & (ms.NNV_d >= cut - pd.Timedelta(days=365))]
    for r in mt.itertuples(): add('Trung bình', 'Master: trạng thái nghỉ nhưng thiếu ngày nghỉ', 'Master Data', '1. TỔNG DATA - NHÂN SỰ', r.Index + 2, r.TEN, f'Trạng thái "{r.TT}", nhận việc {fd(r.NNV_d)} nhưng không có ngày nghỉ hợp lệ ("{r.NNGHI}") → không xếp được vào tuần/tháng nào.', 'Điền ngày nghỉ.')
    ml = ms[(ms.TT == 'Đang làm việc') & _in(ms.NNGHI_d, s, e)]
    for r in ml.itertuples(): add('Cao', 'Master: đang làm việc nhưng có ngày nghỉ', 'Master Data', '1. TỔNG DATA - NHÂN SỰ', r.Index + 2, r.TEN, f'Trạng thái "Đang làm việc" nhưng có ngày nghỉ {fd(r.NNGHI_d)}.', 'Cập nhật trạng thái hoặc xóa ngày nghỉ.')
    # ---- 6) vị trí chưa phân nhóm KTV/TVV
    for col, nm, df in (('VITRI', 'nhận việc (DATA)', nv), ('CV', 'nghỉ việc', ng)):
        if len(df):
            vc = df[col].map(nhom_vi_tri); odd = df[vc == 'Khác'][col].fillna('(trống)').value_counts()
            for v_, n_ in odd.items(): add('Thấp', 'Vị trí không thuộc KTV / TVV', 'DATA / Kế toán / Master', nm, None, str(v_), f'{n_} người {nm} có vị trí/chức vụ "{v_}" – dashboard xếp vào nhóm "Khác".', 'Bình thường nếu là bảo vệ/tạp vụ/văn phòng; nếu là KTV/TVV thì sửa cách viết.')
    out = pd.DataFrame(I, columns=['Mức độ', 'Loại', 'File', 'Sheet', 'Dòng', 'Họ tên', 'Chi tiết', 'Gợi ý xử lý'])
    if len(out):
        out['_o'] = out['Mức độ'].map(SEV_ORDER); out = out.sort_values(['_o', 'Loại', 'Họ tên']).drop(columns='_o').reset_index(drop=True); out.insert(0, 'STT', range(1, len(out) + 1))
    else: out.insert(0, 'STT', [])
    return out

# ------------------------------------------------------------------ PHÂN TÍCH CHUYÊN SÂU (tính từ dòng raw, không gồm điều chỉnh khớp báo cáo)
def _cv(d, s, e): return d[_in(d.NGAYLL_d, s, e)].copy()
def screening(d, s, e, key=None):
    """Sàng lọc CV: kết quả đánh giá CV (Phù hợp / Lưu / Không phù hợp) và đi tiếp tới PV, Pass, nhận việc (theo đợt CV tháng này)."""
    c = _cv(d, s, e)
    if not len(c): return pd.DataFrame()
    c['DG'] = c.DANHGIA.fillna('(chưa đánh giá)'); c['_pv'] = c.KQPV.isin(PVOK); c['_ps'] = c.KQPV == 'Đạt'; c['_nv'] = c.KQNV == 'Đã nhận việc'
    if key is None:
        g = c.groupby('DG').agg(CV=('TEN', 'size'), PV=('_pv', 'sum'), PASS=('_ps', 'sum'), NV=('_nv', 'sum')).astype(int)
        g['% tổng CV'] = [pct(a, len(c)) for a in g.CV]; g['% đi tới PV'] = [pct(a, b) for a, b in zip(g.PV, g.CV)]; g['% Pass'] = [pct(a, b) for a, b in zip(g.PASS, g.CV)]; g['% nhận việc'] = [pct(a, b) for a, b in zip(g.NV, g.CV)]
        return g.sort_values('CV', ascending=False)
    c['_ph'] = c.DG == 'Phù hợp'; c['_lu'] = c.DG == 'Lưu'; c['_kp'] = c.DG == 'Không phù hợp'; c['_dg'] = c.DG != '(chưa đánh giá)'
    c['_ph_pv'] = c._ph & c._pv; c['_ph_nv'] = c._ph & c._nv
    g = c.groupby(key).agg(CV=('TEN', 'size'), DA_DANH_GIA=('_dg', 'sum'), PHU_HOP=('_ph', 'sum'), LUU=('_lu', 'sum'), KHONG_PHU_HOP=('_kp', 'sum'), PH_DEN_PV=('_ph_pv', 'sum'), PH_NV=('_ph_nv', 'sum')).astype(int)
    g['% Phù hợp'] = [pct(a, b) for a, b in zip(g.PHU_HOP, g.DA_DANH_GIA)]; g['% Lưu'] = [pct(a, b) for a, b in zip(g.LUU, g.DA_DANH_GIA)]; g['% Không phù hợp'] = [pct(a, b) for a, b in zip(g.KHONG_PHU_HOP, g.DA_DANH_GIA)]
    g['% Phù hợp đi tới PV'] = [pct(a, b) for a, b in zip(g.PH_DEN_PV, g.PHU_HOP)]; g['% Phù hợp nhận việc'] = [pct(a, b) for a, b in zip(g.PH_NV, g.PHU_HOP)]
    return g.sort_values('CV', ascending=False)

LEAD_BINS = [(0, 1, '0–1 ngày'), (2, 3, '2–3 ngày'), (4, 7, '4–7 ngày'), (8, 14, '8–14 ngày'), (15, 30, '15–30 ngày'), (31, 10**6, 'Trên 30 ngày')]
def lead_times(d, s, e):
    """Thời gian tuyển của người nhận việc trong tháng: liên lạc → PV → nhận việc (ngày)."""
    nv = d[_in(d.NGAYNV_d, s, e) & (d.KQNV == 'Đã nhận việc')].copy(); out = dict(n=len(nv))
    nv['ll_pv'] = (nv.NGAYPV_d - nv.NGAYLL_d).dt.days; nv['pv_nv'] = (nv.NGAYNV_d - nv.NGAYPV_d).dt.days; nv['ll_nv'] = (nv.NGAYNV_d - nv.NGAYLL_d).dt.days
    rows = []
    for k, nm in (('ll_pv', 'Liên lạc → PV'), ('pv_nv', 'PV → nhận việc'), ('ll_nv', 'Liên lạc → nhận việc')):
        v = nv[k]; ok = v[v >= 0]
        rows.append({'Khoảng': nm, 'Số người có đủ ngày': int(v.notna().sum()), 'Hợp lệ (≥0)': int(len(ok)), 'Ngày sai thứ tự': int((v < 0).sum()), 'TB (ngày)': float(ok.mean()) if len(ok) else np.nan, 'Trung vị': float(ok.median()) if len(ok) else np.nan,
                     'P75': float(ok.quantile(.75)) if len(ok) else np.nan, 'Lớn nhất': float(ok.max()) if len(ok) else np.nan})
    out['stats'] = pd.DataFrame(rows).set_index('Khoảng'); ok = nv[nv.ll_nv >= 0]
    dist = [{'Khoảng': lb, 'So_nguoi': int(((ok.ll_nv >= lo) & (ok.ll_nv <= hi)).sum())} for lo, hi, lb in LEAD_BINS]
    dd = pd.DataFrame(dist).set_index('Khoảng'); dd['%'] = [pct(a, len(ok)) for a in dd.So_nguoi]; dd['% cộng dồn'] = dd['%'].cumsum(); out['dist'] = dd
    if len(ok):
        bs = ok.groupby('NGUON').agg(NV=('TEN', 'size'), TB=('ll_nv', 'mean'), Trung_vi=('ll_nv', 'median'), Max=('ll_nv', 'max')).sort_values('NV', ascending=False); bs.columns = ['NV', 'TB (ngày)', 'Trung vị (ngày)', 'Lớn nhất']; out['by_source'] = bs
        bp = ok.groupby('VITRI').agg(NV=('TEN', 'size'), TB=('ll_nv', 'mean'), Trung_vi=('ll_nv', 'median')).sort_values('NV', ascending=False).head(15); bp.columns = ['NV', 'TB (ngày)', 'Trung vị (ngày)']; out['by_pos'] = bp
    else: out['by_source'] = pd.DataFrame(); out['by_pos'] = pd.DataFrame()
    return out

def canon_person(x):
    """Gộp cách viết tên người phỏng vấn (A Pháp / a Pháp / Pháp -> PHAP)."""
    if x is None or (isinstance(x, float) and pd.isna(x)): return '(không ghi)'
    t = norm(x).split()
    while len(t) > 1 and t[0] in ('A', 'C', 'ANH', 'CHI', 'E', 'CO', 'EM'): t = t[1:]
    return ' '.join(t) or '(không ghi)'
def interviewers(d, s, e):
    x = d[_in(d.NGAYPV_d, s, e) & (d.KQPV.isin(PVOK) | (d.KQPV == 'Không đến pv'))].copy()
    if not len(x): return pd.DataFrame()
    x['NGUOI'] = x.NGUOIPV.map(canon_person); x['_den'] = x.KQPV.isin(PVOK); x['_ps'] = x.KQPV == 'Đạt'; x['_kd'] = x.KQPV == 'Không đến pv'; x['_nv'] = (x.KQPV == 'Đạt') & (x.KQNV == 'Đã nhận việc')
    g = x.groupby('NGUOI').agg(LICH_HEN=('TEN', 'size'), KHONG_DEN=('_kd', 'sum'), THAM_GIA=('_den', 'sum'), PASS=('_ps', 'sum'), NV=('_nv', 'sum')).astype(int)
    g['% Không đến'] = [pct(a, b) for a, b in zip(g.KHONG_DEN, g.LICH_HEN)]; g['% Pass / tham gia'] = [pct(a, b) for a, b in zip(g.PASS, g.THAM_GIA)]; g['% Nhận việc / Pass'] = [pct(a, b) for a, b in zip(g.NV, g.PASS)]
    return g.sort_values('THAM_GIA', ascending=False)

def crosstab_conv(d, s, e, rowk, colk, nrow=12, ncol=8):
    """Ma trận (đợt CV tháng này): số CV, số nhận việc và % nhận việc / CV theo hai chiều."""
    c = _cv(d, s, e)
    if not len(c): return None
    top_r = c[rowk].value_counts().head(nrow).index; top_c = c[colk].value_counts().head(ncol).index
    c['_r'] = c[rowk].where(c[rowk].isin(top_r), 'Khác'); c['_c'] = c[colk].where(c[colk].isin(top_c), 'Khác'); c['_nv'] = (c.KQNV == 'Đã nhận việc').astype(int)
    cv = c.pivot_table(index='_r', columns='_c', values='TEN', aggfunc='count', fill_value=0); nv = c.pivot_table(index='_r', columns='_c', values='_nv', aggfunc='sum', fill_value=0)
    cv['Tổng'] = cv.sum(axis=1); nv['Tổng'] = nv.sum(axis=1); cv.loc['Tổng'] = cv.sum(); nv.loc['Tổng'] = nv.sum()
    rate = (nv / cv.replace(0, np.nan) * 100)
    order = [r for r in cv.index if r != 'Tổng'] ; order = sorted(order, key=lambda r: -cv.loc[r, 'Tổng']) + ['Tổng']
    return dict(cv=cv.loc[order].astype(int), nv=nv.loc[order].astype(int), rate=rate.loc[order])

def parse_luong(x):
    """'5tr' -> 5.0 ; 'cth 12tr' -> 12 ; '11tr (24h)' -> 11 ; '5,5tr' -> 5.5 ; số >=100000 coi là VND."""
    if x is None or (isinstance(x, float) and pd.isna(x)): return np.nan
    t = str(x).lower().replace(',', '.')
    m = re.search(r'(\d+(?:\.\d+)?)\s*(?:tr|triệu|trieu|m)\b', t) or re.search(r'(\d+(?:\.\d+)?)\s*(?:tr|triệu|trieu)', t)
    if m: return float(m.group(1))
    m = re.fullmatch(r'\s*(\d{6,9})\s*', t)
    return float(m.group(1)) / 1e6 if m else np.nan
def salary_table(d, s, e):
    c = _cv(d, s, e)
    if not len(c): return dict(cover=0, n=0, table=pd.DataFrame())
    c['LUONG_TR'] = c.LUONG.map(parse_luong); c['_nv'] = c.KQNV == 'Đã nhận việc'; x = c[c.LUONG_TR.notna()]
    out = dict(cover=pct(len(x), len(c)), n=len(x), cv=len(c))
    if not len(x): out['table'] = pd.DataFrame(); return out
    g = x.groupby(x.VITRI.fillna('(không ghi)')).agg(SO_CV_CO_LUONG=('TEN', 'size'), TRUNG_VI=('LUONG_TR', 'median'), TB=('LUONG_TR', 'mean'), THAP_NHAT=('LUONG_TR', 'min'), CAO_NHAT=('LUONG_TR', 'max'), NV=('_nv', 'sum'))
    g['NV'] = g.NV.astype(int); g['% nhận việc'] = [pct(a, b) for a, b in zip(g.NV, g.SO_CV_CO_LUONG)]; g.columns = ['Số CV có ghi lương', 'Trung vị (triệu)', 'TB (triệu)', 'Thấp nhất', 'Cao nhất', 'Nhận việc', '% nhận việc']
    out['table'] = g.sort_values('Số CV có ghi lương', ascending=False).head(15); out['overall'] = dict(median=float(x.LUONG_TR.median()), mean=float(x.LUONG_TR.mean()))
    return out

RET_DAYS = [7, 14, 30, 60, 90]
def retention_detail(nv, cutoff):
    """Đường cong giữ chân của người nhận việc trong tháng: % còn làm sau 7/14/30/60/90 ngày (chỉ tính người đã đủ ngày tính tới ngày dữ liệu mới nhất)."""
    out = {}; rows = []
    for t in RET_DAYS:
        mat = nv[nv.NNV + pd.Timedelta(days=t) <= cutoff]; q = int((mat.NGAY_DEN_KHI_NGHI <= t).sum())
        rows.append({'Mốc': f'Sau {t} ngày', 'Đã đủ ngày': len(mat), 'Đã nghỉ trong mốc': q, '% còn làm': (100 - pct(q, len(mat))) if len(mat) else np.nan, '% đã nghỉ': pct(q, len(mat))})
    out['curve'] = pd.DataFrame(rows).set_index('Mốc')
    mat = nv[nv.NNV + pd.Timedelta(days=30) <= cutoff]
    for key, nm in (('NGUON', 'by_source'), ('CN', 'by_unit'), ('VITRI', 'by_pos'), ('LH', 'by_hr')):
        if len(mat):
            g = mat.groupby(mat[key].fillna('(không ghi)')).agg(DA_DU_30_NGAY=('TEN', 'size'), NGHI_30=('NGAY_DEN_KHI_NGHI', lambda x: int((x <= 30).sum()))); g['% nghỉ trong 30 ngày'] = [pct(a, b) for a, b in zip(g.NGHI_30, g.DA_DU_30_NGAY)]
            out[nm] = g.sort_values('DA_DU_30_NGAY', ascending=False)
        else: out[nm] = pd.DataFrame()
    return out

def trend_year(raw, year):
    """Xu hướng 12 tháng của một năm (từ raw): phễu, nghỉ việc, tỷ lệ chuyển đổi và nguồn."""
    d = raw['d']; nga = raw['nghi_all']; rows = []; src_cv = {}; src_nv = {}
    for m in range(1, 13):
        s, e = month_range(year, m); pvm = _in(d.NGAYPV_d, s, e); nvm = _in(d.NGAYNV_d, s, e); cvm = _in(d.NGAYLL_d, s, e)
        T = dict(CV=int(cvm.sum()), PV=int((pvm & d.KQPV.isin(PVOK)).sum()), PASS=int((pvm & (d.KQPV == 'Đạt')).sum()), KHONG_DEN_PV=int((pvm & (d.KQPV == 'Không đến pv')).sum()),
                 NV=int((nvm & (d.KQNV == 'Đã nhận việc')).sum()), KHONG_NV=int((nvm & (d.KQNV == 'Không nhận việc')).sum()))
        ng = nga[_in(nga.NNGHI, s, e)] if len(nga) else nga
        T['NGHI'] = len(ng); T['NGHI_TV'] = int((ng.days < 30).sum()) if len(ng) else 0; T['NGHI_SOM'] = int((ng.days <= 10).sum()) if len(ng) else 0; T['THANG'] = m
        rows.append(T); src_cv[m] = d[cvm].groupby('NGUON').size(); src_nv[m] = d[nvm & (d.KQNV == 'Đã nhận việc')].groupby('NGUON').size()
    t = pd.DataFrame(rows).set_index('THANG'); t.index = [f'T{m:02d}' for m in t.index]
    t['% PV/CV'] = [pct(a, b) for a, b in zip(t.PV, t.CV)]; t['% Pass/PV'] = [pct(a, b) for a, b in zip(t.PASS, t.PV)]; t['% NV/Pass'] = [pct(a, b) for a, b in zip(t.NV, t.PASS)]; t['% NV/CV'] = [pct(a, b) for a, b in zip(t.NV, t.CV)]
    t['% Nghỉ/Nhận việc'] = [pct(a, b) for a, b in zip(t.NGHI, t.NV)]; t['% Nghỉ thử việc/Nghỉ'] = [pct(a, b) for a, b in zip(t.NGHI_TV, t.NGHI)]; t['Tăng ròng (NV − Nghỉ)'] = t.NV - t.NGHI
    cnt = ['CV', 'PV', 'PASS', 'NV', 'NGHI']
    mom = t[cnt].pct_change().replace([np.inf, -np.inf], np.nan) * 100; mom.columns = [f'Δ% {c} so với tháng trước' for c in cnt]
    scv = pd.DataFrame(src_cv).fillna(0).astype(int); snv = pd.DataFrame(src_nv).fillna(0).astype(int)
    top = snv.sum(axis=1).sort_values(ascending=False).head(6).index
    scv.columns = [f'T{m:02d}' for m in scv.columns]; snv.columns = [f'T{m:02d}' for m in snv.columns]
    return dict(table=t, mom=mom, src_cv=scv.loc[scv.sum(axis=1).sort_values(ascending=False).head(8).index], src_nv=snv.loc[top])

def insights(R, deep):
    """Điểm nổi bật tự động (mỗi câu đều kèm số liệu và cỡ mẫu)."""
    T = R['funnel']; out = []
    if T['CV'] == 0: return out
    steps = [('Tham gia PV / CV', pct(T['PV'], T['CV'])), ('Đạt PV / tham gia PV', pct(T['PASS'], T['PV'])), ('Nhận việc / Đạt PV', pct(T['NV'], T['PASS']))]
    out.append(f"Phễu: {T['CV']:,} CV → {T['PV']:,} tham gia PV → {T['PASS']:,} đạt → {T['NV']:,} nhận việc ({pct(T['NV'], T['CV']):.1f}% CV thành nhận việc). Bước giữ lại ít nhất: **{min(steps, key=lambda x: (x[1] if x[1] == x[1] else 999))[0]}** ({min(steps, key=lambda x: (x[1] if x[1] == x[1] else 999))[1]:.1f}%).")
    if T['KHONG_DEN_PV'] and T['PV']: out.append(f"{T['KHONG_DEN_PV']:,} lượt hẹn không đến PV ({pct(T['KHONG_DEN_PV'], T['PV'] + T['KHONG_DEN_PV']):.1f}% số hẹn).")
    sc = R['by_source']; big = sc[sc.CV >= max(30, 0.05 * T['CV'])]
    if len(big):
        b = big.sort_values('%NV/CV', ascending=False).iloc[0]; w = big.sort_values('%NV/CV').iloc[0]
        out.append(f"Nguồn nhận việc hiệu quả nhất (≥{max(30, int(0.05 * T['CV']))} CV): **{b.name}** – {int(b.NV)} nhận việc / {int(b.CV)} CV ({b['%NV/CV']:.1f}%). Thấp nhất: **{w.name}** – {int(w.NV)}/{int(w.CV)} ({w['%NV/CV']:.1f}%).")
        z = big[(big.PASS >= 3) & (big.NV == 0)]
        if len(z): out.append('Nguồn có từ 3 Pass nhưng chưa có nhận việc: ' + ', '.join(f"{i} ({int(r.PASS)} Pass)" for i, r in z.iterrows()) + '.')
    if R['nghi_total'] and T['NV']:
        ratio = pct(R['nghi_total'], T['NV'])
        rtxt = f"tương đương {ratio:.1f}% số nhận việc" if ratio <= 100 else f"nhiều hơn số nhận việc trong tháng ({T['NV']:,}) vì gồm cả người vào làm từ các tháng trước"
        out.append(f"Nghỉ việc {R['nghi_total']:,} người, {rtxt}; {pct(R['nghi_tv'], R['nghi_total']):.1f}% là nghỉ thử việc (<30 ngày) và {pct(R['nghi_early10'], R['nghi_total']):.1f}% nghỉ trong 10 ngày đầu.")
    ru = R['nghi_unit']
    if len(ru):
        r0 = ru.iloc[0]; out.append(f"Chi nhánh/bộ phận nghỉ nhiều nhất: **{ru.index[0]}** ({int(r0.NGHI)} người, {r0['%Tỷ trọng']:.1f}% tổng nghỉ).")
    sc_ = deep.get('screen')
    if sc_ is not None and len(sc_) and 'Phù hợp' in sc_.index:
        out.append(f"Sàng lọc CV: {sc_.loc['Phù hợp', 'CV']:,} CV 'Phù hợp' ({sc_.loc['Phù hợp', '% tổng CV']:.1f}% tổng CV), trong đó {sc_.loc['Phù hợp', '% đi tới PV']:.1f}% đi tới PV.")
    lt = deep.get('lead')
    if lt and lt['stats'].loc['Liên lạc → nhận việc', 'Hợp lệ (≥0)'] > 0: out.append(f"Thời gian từ liên lạc đến nhận việc: trung vị {lt['stats'].loc['Liên lạc → nhận việc', 'Trung vị']:.0f} ngày, trung bình {lt['stats'].loc['Liên lạc → nhận việc', 'TB (ngày)']:.1f} ngày (P75 {lt['stats'].loc['Liên lạc → nhận việc', 'P75']:.0f} ngày).")
    rt = deep.get('ret')
    if rt is not None and rt['curve'].loc['Sau 30 ngày', 'Đã đủ ngày'] > 0: out.append(f"Giữ chân: trong {int(rt['curve'].loc['Sau 30 ngày', 'Đã đủ ngày'])} người đã đủ 30 ngày, {rt['curve'].loc['Sau 30 ngày', '% đã nghỉ']:.1f}% đã nghỉ trong 30 ngày đầu.")
    return out

def analyze_deep(raw, R):
    d = raw['d']; s, e = R['start'], R['end']
    return dict(screen=screening(d, s, e), screen_src=screening(d, s, e, 'NGUON'), screen_hr=screening(d, s, e, 'LH'), lead=lead_times(d, s, e), interv=interviewers(d, s, e),
                x_src_unit=crosstab_conv(d, s, e, 'NGUON', 'DON_VI'), x_src_pos=crosstab_conv(d, s, e, 'NGUON', 'VITRI'), salary=salary_table(d, s, e), ret=retention_detail(R['nv_list'], raw['cutoff']))

# =====================================================================================================
# 3) GIAO DIỆN STREAMLIT
# =====================================================================================================
MONTHS = {i: f'Tháng {i:02d}' for i in range(1, 13)}

@st.cache_resource(show_spinner='Đang đọc 4 file raw (khoảng 15–30 giây)…')
def load_folder(folder, sig):
    f = find_files(folder); need = [k for k in ('data', 'kt', 'master') if k not in f]
    if need: raise FileNotFoundError('Thiếu file trong thư mục: ' + ', '.join(need))
    return load_raw(f['data'], f['kt'], f['master'], f.get('ads')), {k: os.path.basename(v) for k, v in f.items()}
@st.cache_data(show_spinner=False)
def classify_cached(b):
    return classify_file(b)
@st.cache_resource(show_spinner='Đang đọc file đã tải lên…')
def load_bytes(a, b, c, d, sig):
    return load_raw(a, b, c, d), {}

def colcfg(df):
    cfg = {}
    for c in df.columns:
        if str(c).startswith('%') or str(c).startswith('Δ%'): cfg[c] = st.column_config.NumberColumn(c, format='%.1f%%')
        elif str(c).startswith(('Chi phí', 'Tổng chi')): cfg[c] = st.column_config.NumberColumn(c, format='%.0f')
    return cfg
def show(df, **kw):
    if df is None or len(df) == 0: st.caption('Không có dữ liệu.'); return
    st.dataframe(df, column_config=colcfg(df), width='stretch', **kw)
def fmt_pct(x): return '–' if pd.isna(x) else f'{x:.1f}%'
def fmt_vnd(x): return '–' if pd.isna(x) else f'{x:,.0f}'



def main():
    st.set_page_config(page_title='Báo cáo tuyển dụng theo tháng', layout='wide')
    # ------------------------------------------------------------------ thanh bên
    st.sidebar.header('Dữ liệu')
    mode = st.sidebar.radio('Nguồn dữ liệu', ['Tải file lên', 'Thư mục trên máy'], index=(1 if os.environ.get('TD_DATA_DIR') else 0), key='mode')
    raw = None; names = {}
    if mode == 'Thư mục trên máy':
        folder = st.sidebar.text_input('Đường dẫn thư mục chứa các file raw', value=os.environ.get('TD_DATA_DIR', ''), key='folder')
        if folder:
            try:
                sig = tuple(sorted((k, os.path.getmtime(v)) for k, v in find_files(folder).items()))
                raw, names = load_folder(folder, sig)
            except Exception as ex: st.sidebar.error(f'Không đọc được: {ex}')
    else:
        files = st.sidebar.file_uploader('Kéo thả các file raw vào đây (3–4 file: DATA tuyển dụng, Kế toán Admin, Master Data, Report Facebook Ads). Thứ tự và tên file tùy ý – app tự nhận ra theo nội dung.', type=['xlsx'], accept_multiple_files=True, key='up_all')
        groups = {}; unknown = []
        for f_ in (files or []):
            b_ = f_.getvalue(); kind_, why_ = classify_cached(b_)
            if kind_: groups.setdefault(kind_, []).append((getattr(f_, 'name', kind_), b_))
            else: unknown.append((getattr(f_, 'name', '?'), why_))
        chosen = {}
        for kind_, lst in groups.items():
            if len(lst) > 1:
                opts = [f'{i_ + 1}. {n_}' for i_, (n_, _) in enumerate(lst)]
                st.sidebar.warning(f'Có {len(lst)} file {KIND_LABEL[kind_]} – chọn file cần dùng:')
                pick_ = st.sidebar.selectbox('Chọn file ' + KIND_LABEL[kind_], opts, index=len(opts) - 1, key=f'pick_{kind_}')
                chosen[kind_] = lst[opts.index(pick_)]
            else: chosen[kind_] = lst[0]
        for kind_ in ('data', 'kt', 'master', 'ads'):
            if kind_ in chosen: st.sidebar.caption(f'✅ {KIND_LABEL[kind_]}: {chosen[kind_][0]}')
        for n_, why_ in unknown: st.sidebar.warning(f'Bỏ qua file "{n_}": {why_}.')
        miss_ = [KIND_LABEL[k_] for k_ in ('data', 'kt', 'master') if k_ not in chosen]
        if miss_: st.sidebar.info('Còn thiếu: ' + ', '.join(miss_) + '.')
        else:
            try:
                raw, names = load_bytes(chosen['data'][1], chosen['kt'][1], chosen['master'][1], chosen['ads'][1] if 'ads' in chosen else None, tuple(len(v_[1]) for v_ in chosen.values()))
                names = {k_: v_[0] for k_, v_ in chosen.items()}
            except Exception as ex: st.sidebar.error(f'Không đọc được file đã tải lên: {ex}')
    if raw is None:
        st.title('Báo cáo tuyển dụng theo tháng')
        if mode == 'Tải file lên': st.info('👈 Kéo thả 3–4 file raw vào ô "Kéo thả các file raw" ở thanh bên trái: DATA tuyển dụng, Kế toán Admin Update, Master Data (bắt buộc) và Report Facebook Ads (không bắt buộc). Tên file và thứ tự tùy ý – app tự nhận ra.')
        else: st.info('👈 Nhập đường dẫn thư mục chứa các file raw ở thanh bên trái, hoặc đổi sang "Tải file lên" để kéo thả file.')
        st.stop()

    years = sorted({int(y) for y in raw['d'].NGAYLL_d.dropna().dt.year.unique() if 2020 <= y <= 2100}) or [2026]
    year = st.sidebar.selectbox('Năm', years, index=len(years) - 1, key='year')
    sel = st.sidebar.multiselect('Chọn tháng cần báo cáo', list(MONTHS), default=[9, 11, 12], format_func=lambda m: MONTHS[m], key='months')
    xl_slot = st.sidebar.container()
    if names: st.sidebar.caption('File đang dùng: ' + '; '.join(f'{v}' for v in names.values()))
    st.sidebar.caption(f"Ngày dữ liệu mới nhất: {raw['cutoff']:%d/%m/%Y}")
    if not sel: st.title('Báo cáo tuyển dụng theo tháng'); st.warning('Chọn ít nhất một tháng.'); st.stop()
    sel = sorted(sel)
    has_adj = [m for m in sel if get_adj(year, m)]
    use_adj = st.sidebar.checkbox('Khớp báo cáo đã duyệt (áp dụng điều chỉnh)', value=True, key='use_adj',
                                  help='Tháng có điều chỉnh: ' + (', '.join(f'{m:02d}/{year}' for m in sorted(k[1] for k in ADJ if k[0] == year)) or 'không có') + '. Tắt để xem số thuần từ raw.')
    if use_adj and has_adj: st.sidebar.caption('Đang áp dụng điều chỉnh cho: ' + ', '.join(f'T{m:02d}' for m in has_adj))
    elif not use_adj: st.sidebar.caption('Đang xem số thuần từ raw (chưa điều chỉnh).')
    use_range = st.sidebar.checkbox('Xem thêm theo khoảng ngày tùy chọn', value=False, key='use_range', help='Giống dashboard theo dõi: chọn Từ ngày – Đến ngày bất kỳ.')
    rng = None
    if use_range:
        cut_ = raw['cutoff']; d0 = (cut_ - pd.Timedelta(days=cut_.day - 1)).date()
        v_ = st.sidebar.date_input('Từ ngày – đến ngày', value=(d0, cut_.date()), key='rng')
        if isinstance(v_, (tuple, list)) and len(v_) == 2 and v_[0] <= v_[1]: rng = (pd.Timestamp(v_[0]), pd.Timestamp(v_[1]))
        else: st.sidebar.info('Chọn đủ ngày bắt đầu và ngày kết thúc.')
    manual = {}
    with st.sidebar.expander('Chi phí nhập tay (không có trong raw)'):
        for m in sel:
            st.markdown(f'**{MONTHS[m]}**')
            ad_ = (get_adj(year, m) if use_adj else None) or {}; cp_ = ad_.get('chi_phi_tay', {}); fl = int(bool(ad_))
            manual[m] = {'CareerViet': st.number_input('CareerViet (VND)', min_value=0.0, step=100000.0, value=float(cp_.get('CareerViet', 0.0)), key=f'cv_{m}_{fl}'),
                         'TopCV': st.number_input('TopCV (VND)', min_value=0.0, step=100000.0, value=float(cp_.get('TopCV', 0.0)), key=f'tc_{m}_{fl}'),
                         'Khác': st.number_input('Chi phí khác (VND)', min_value=0.0, step=100000.0, value=0.0, key=f'ot_{m}_{fl}')}

    if 'res' not in st.session_state: st.session_state['res'] = {}
    def get_result(m):
        adj = get_adj(year, m) if use_adj else None
        key = (id(raw), year, m, tuple(sorted(manual[m].items())), bool(adj))
        if key not in st.session_state['res']:
            st.session_state['res'][key] = compute_month(raw, year, m, {k: v for k, v in manual[m].items() if v}, adj)
        return st.session_state['res'][key]
    results = [get_result(m) for m in sel]
    def get_deep(R):
        key = ('deep', id(raw), R['start'], R['end'], R['adjusted'])
        if key not in st.session_state['res']: st.session_state['res'][key] = analyze_deep(raw, R)
        return st.session_state['res'][key]
    def get_weekly(m):
        key = ('weekly', id(raw), year, m)
        if key not in st.session_state['res']: st.session_state['res'][key] = weekly_summary(raw, year, m)
        return st.session_state['res'][key]
    def get_checks(R):
        key = ('check', id(raw), R['start'], R['end'])
        if key not in st.session_state['res']: st.session_state['res'][key] = kiem_tra_du_lieu(raw, R['start'], R['end'])
        return st.session_state['res'][key]
    def get_period(s_, e_, label, m):
        key = ('period', id(raw), s_, e_)
        if key not in st.session_state['res']: st.session_state['res'][key] = compute_month(raw, year, m, None, None, period=(s_, e_), label=label)
        return st.session_state['res'][key]
    def get_trend():
        key = ('trend', id(raw), year)
        if key not in st.session_state['res']: st.session_state['res'][key] = trend_year(raw, year)
        return st.session_state['res'][key]

    st.title('Báo cáo tuyển dụng theo tháng')
    st.caption('Định nghĩa: CV = ngày liên lạc trong tháng · Tham gia PV = ngày PV trong tháng (Đạt/Đợi KQ/Không đạt) · Pass = ngày PV trong tháng & Đạt · Nhận việc = ngày nhận việc trong tháng & "Đã nhận việc" · Nghỉ việc = hợp Kế toán + Master theo ngày nghỉ.')

    bt_store = {}

    fixed_tabs = ['📊 So sánh các tháng', f'📈 Xu hướng năm {year}', '📅 Theo tuần', '🔎 Kiểm tra dữ liệu'] + (['🗓 Khoảng ngày'] if rng else [])
    NF = len(fixed_tabs); tabs = st.tabs(fixed_tabs + [MONTHS[m] for m in sel])
    # ------------------------------------------------------------------ so sánh
    with tabs[0]:
        cmp_ = compare(results)
        st.subheader('So sánh chỉ tiêu chính')
        st.caption('Cột Δ% = % thay đổi giữa hai tháng liền nhau (chỉ tính cho các dòng số lượng/tiền, không tính cho dòng đã là %).')
        show(cmp_)
        cnt = cmp_.loc[['Số CV (liên lạc)', 'Tham gia PV', 'Đạt PV (Pass)', 'Nhận việc', 'Nghỉ việc'], [c for c in cmp_.columns if not c.startswith('Δ')]]
        st.markdown('**Số lượng theo tháng**'); st.bar_chart(cnt.T)
        pc = cmp_.loc[['% PV / CV', '% Pass / PV', '% Nhận việc / Pass', '% Nhận việc / CV'], [c for c in cmp_.columns if not c.startswith('Δ')]]
        st.markdown('**Tỷ lệ chuyển đổi (%)**'); st.line_chart(pc.T)
        empty = [f"{MONTHS[R['month']]}" for R in results if not R['has_data']]
        if empty: st.warning('Chưa có dữ liệu tuyển dụng cho: ' + ', '.join(empty) + f". Ngày dữ liệu mới nhất trong file: {raw['cutoff']:%d/%m/%Y}.")

    # ------------------------------------------------------------------ từng tháng
    with tabs[1]:
        TR = get_trend(); tt = TR['table']
        st.subheader(f'Xu hướng 12 tháng năm {year}')
        st.caption('Tính từ dòng raw (không gồm điều chỉnh khớp báo cáo). Tháng chưa có dữ liệu hiện bằng 0.')
        cols_n = ['CV', 'PV', 'PASS', 'KHONG_DEN_PV', 'NV', 'KHONG_NV', 'NGHI', 'NGHI_TV', 'NGHI_SOM', 'Tăng ròng (NV − Nghỉ)']
        st.markdown('**Số lượng theo tháng**'); show(tt[cols_n])
        st.markdown('**Tỷ lệ chuyển đổi và nghỉ việc (%)**'); show(tt[['% PV/CV', '% Pass/PV', '% NV/Pass', '% NV/CV', '% Nghỉ/Nhận việc', '% Nghỉ thử việc/Nghỉ']])
        l, r_ = st.columns(2)
        with l: st.markdown('**CV, tham gia PV, Pass, nhận việc**'); st.bar_chart(tt[['CV', 'PV', 'PASS', 'NV']])
        with r_: st.markdown('**Nhận việc so với nghỉ việc**'); st.bar_chart(tt[['NV', 'NGHI']])
        st.markdown('**Tỷ lệ chuyển đổi theo tháng (%)**'); st.line_chart(tt[['% PV/CV', '% Pass/PV', '% NV/Pass', '% NV/CV']])
        st.markdown('**Thay đổi so với tháng trước (%)**'); show(TR['mom'])
        l, r_ = st.columns(2)
        with l: st.markdown('**Nhận việc theo nguồn (6 nguồn nhiều nhất)**'); show(TR['src_nv']); st.line_chart(TR['src_nv'].T)
        with r_: st.markdown('**CV theo nguồn (8 nguồn nhiều nhất)**'); show(TR['src_cv']); st.line_chart(TR['src_cv'].T)
    def render_period(R, K):
        """Hiển thị đầy đủ một kỳ (tháng, tuần hoặc khoảng ngày bất kỳ)."""
        T = R['funnel']; c = R['cost']
        st.header(f"{R['label']}  ({R['start']:%d/%m} – {R['end']:%d/%m})")
        if not R['has_data']: st.warning('Không có CV hoặc nhận việc nào trong kỳ này – có thể file chưa cập nhật tới kỳ này.')
        elif R['end'] > R['cutoff']: st.info(f"Kỳ chưa trọn: dữ liệu mới nhất đến {R['cutoff']:%d/%m/%Y}. Số liệu còn thay đổi.")
        if R['adjusted']:
            ad = R['adj']; st.success(f"Đã khớp báo cáo đã duyệt: {ad.get('ten', '')}. Số liệu dưới đây gồm các điều chỉnh liệt kê trong khung bên dưới (tắt ở thanh bên để xem số thuần từ raw).")
            if R['adj_baseline_diff']:
                nm_ = {'CV': 'số CV', 'PV': 'tham gia PV', 'PASS': 'Pass', 'KHONG_DAT': 'không đạt', 'DOI_KQ': 'đợi kết quả', 'KHONG_DEN_PV': 'không đến PV', 'NV': 'nhận việc', 'KHONG_NV': 'không nhận việc', 'NGHI': 'nghỉ việc'}
                st.warning('⚠️ File raw hiện tại khác bản raw dùng để lập các điều chỉnh: ' + '; '.join(f'{nm_.get(k_, k_)} = {a_} (lúc lập điều chỉnh: {b_})' for k_, (a_, b_) in R['adj_baseline_diff'].items()) + '. Các điều chỉnh có thể không còn đúng – kiểm tra lại hoặc tắt "Khớp báo cáo đã duyệt".')
            with st.expander('Xem các điều chỉnh đã áp dụng và lý do'):
                for x in ad.get('nhan_viec_bo_sung', []): st.markdown(f"- **Thêm người nhận việc** {x['TEN']} ({x['CN']}, {x['NNV']}): {x['ly_do']}")
                for k_, dt_, why in ad.get('loai_tru_nghi', []): st.markdown(f"- **Không tính vào nghỉ việc** {k_.title()} (nhận việc {dt_}): {why}")
                ly = ad.get('ly_do_delta', {})
                for nm, key in (('Theo nguồn', 'delta_nguon'), ('Theo HR', 'delta_hr'), ('Theo chi nhánh (mọi vị trí)', 'delta_unit'), ('Bảng phòng ban/chi nhánh của báo cáo', 'delta_dept')):
                    if ad.get(key): st.markdown(f"- **{nm}:** " + '; '.join(f'{g_} – {mm}: {v:+d}' for (g_, mm), v in ad[key].items()) + f"  \n  _{ly.get(key.split('_')[1], '')}_")
                st.markdown('- **Nhập tay trong báo cáo:** đầu kỳ, số lượng yêu cầu, chi phí CareerViet/TopCV, Facebook Ads 30–31/10, nhóm lý do nghỉ (phân loại tay).')
        a = st.columns(6)
        a[0].metric('Số CV', f"{T['CV']:,}"); a[1].metric('Tham gia PV', f"{T['PV']:,}", fmt_pct(pct(T['PV'], T['CV'])) + ' của CV', delta_color='off')
        a[2].metric('Pass PV', f"{T['PASS']:,}", fmt_pct(pct(T['PASS'], T['PV'])) + ' của PV', delta_color='off'); a[3].metric('Nhận việc', f"{T['NV']:,}", fmt_pct(pct(T['NV'], T['PASS'])) + ' của Pass', delta_color='off')
        a[4].metric('Nghỉ việc', f"{R['nghi_total']:,}", fmt_pct(pct(R['nghi_total'], T['NV'])) + ' / nhận việc', delta_color='off'); a[5].metric('Chi phí / người nhận việc', fmt_vnd(c['per_nv']))

        Dp = get_deep(R); ins = insights(R, Dp)
        if ins:
            st.subheader('Điểm nổi bật')
            for line in ins: st.markdown('- ' + line)
        st.subheader('1. Phễu tuyển dụng')
        fun = pd.DataFrame({'Bước': ['CV (liên lạc)', 'Tham gia PV', 'Đạt PV', 'Nhận việc'], 'Số lượng': [T['CV'], T['PV'], T['PASS'], T['NV']]})
        fun['% so với bước trước'] = [np.nan] + [pct(fun['Số lượng'][i], fun['Số lượng'][i - 1]) for i in range(1, 4)]; fun['% so với CV'] = [pct(x, T['CV']) for x in fun['Số lượng']]
        fun = fun.rename(columns={'% so với bước trước': '%so với bước trước', '% so với CV': '%so với CV'})
        l, r_ = st.columns([3, 2])
        with l: show(fun.set_index('Bước'))
        with r_: st.bar_chart(fun.set_index('Bước')['Số lượng'])
        st.caption(f"Không đạt PV {T['KHONG_DAT']:,} ({fmt_pct(pct(T['KHONG_DAT'], T['PV']))} của PV) · Không đến PV {T['KHONG_DEN_PV']:,} ({fmt_pct(pct(T['KHONG_DEN_PV'], T['PV'] + T['KHONG_DEN_PV']))} số hẹn) · Đợi kết quả {T['DOI_KQ']:,} · Không nhận việc {T['KHONG_NV']:,} ({fmt_pct(pct(T['KHONG_NV'], T['NV'] + T['KHONG_NV']))} trên số được mời nhận việc).")

        def block(title, df, key, extra=''):
            st.subheader(title); 
            view = st.radio('Cách tính', ['Theo kỳ (như báo cáo tháng)', 'Theo đợt CV của tháng (cohort)'], horizontal=True, key=f'{K}_{key}_view')
            if view.startswith('Theo kỳ'): cols = ['CV', 'PV', 'PASS', 'NV', 'KHONG_NV', '%PV/CV', '%Pass/PV', '%NV/Pass', '%NV/CV', '%Không đến PV', '%Tỷ trọng CV', '%Tỷ trọng NV']
            else: cols = ['CV_DOT', 'DOT_PV', 'DOT_PASS', 'DOT_NV', '%Đợt CV→PV', '%Đợt CV→Pass', '%Đợt CV→NV']
            show(df[cols]); 
            if extra: st.caption(extra)
            top = df.head(12)[['CV', 'PASS', 'NV']] if view.startswith('Theo kỳ') else df.head(12)[['CV_DOT', 'DOT_PASS', 'DOT_NV']]
            st.bar_chart(top)
        st.markdown('---')
        block('2. Theo nguồn CV', R['by_source'], 'src', '"Theo kỳ": mỗi bước tính theo ngày của chính bước đó nên % có thể vượt 100%. "Theo đợt CV": chỉ xét những người liên lạc trong tháng – tỷ lệ luôn ≤100%. Nguồn có ít CV chỉ mang tính tham khảo.')
        block('3. Theo nhân sự tuyển dụng (HR)', R['by_hr'], 'hr')
        block('4. Theo chi nhánh / phòng ban', R['by_unit'], 'unit', 'Tính theo cột Chi nhánh trong DATA, gồm mọi vị trí (báo cáo tháng 10 chỉ lọc một số vị trí nên số có thể lớn hơn).')
        st.subheader('4b. Theo khối'); show(R['by_khoi'][['CV', 'PV', 'PASS', 'NV', '%PV/CV', '%Pass/PV', '%NV/Pass', '%NV/CV']])
        st.subheader('5. Theo vị trí tuyển (top 15 theo CV)'); show(R['by_pos'].sort_values('CV', ascending=False).head(15)[['CV', 'PV', 'PASS', 'NV', '%PV/CV', '%Pass/PV', '%NV/Pass', '%NV/CV']])

        st.markdown('---'); st.subheader('5b. Biến động nhân sự theo chi nhánh (bố cục báo cáo)')
        A_ = R['adj'] if R['adjusted'] else {}
        dk0 = pd.DataFrame({'Đầu kỳ': [A_.get('dau_ky', {}).get(b_) for b_ in BRANCHES], 'SL yêu cầu': [A_.get('sl_yc', {}).get(b_) for b_ in BRANCHES]}, index=BRANCHES, dtype='float')
        st.caption('Đầu kỳ và Số lượng yêu cầu không có trong raw – nhập vào bảng (tháng 10 đã điền sẵn theo báo cáo).')
        ed = st.data_editor(dk0, key=f'{K}_dk_{int(R["adjusted"])}', width='stretch')
        bt = branch_table(R, {k_: v_ for k_, v_ in ed['Đầu kỳ'].items() if v_ == v_}, {k_: v_ for k_, v_ in ed['SL yêu cầu'].items() if v_ == v_}); (bt_store.__setitem__(R['month'], bt) if R['is_month'] else None); show(bt)
        st.subheader('5c. Bảng phòng ban / chi nhánh – đúng định nghĩa báo cáo')
        dr = R['dept_report']
        if len(dr):
            dd = dr.copy(); dd.loc['% Tham gia PV / CV'] = [pct(a_, b_) for a_, b_ in zip(dr.loc['Tham gia PV'], dr.loc['Số CV'])]; dd.loc['% Đạt PV / Tham gia PV'] = [pct(a_, b_) for a_, b_ in zip(dr.loc['Đạt PV'], dr.loc['Tham gia PV'])]; dd.loc['% Đã NV / Đạt PV'] = [pct(a_, b_) for a_, b_ in zip(dr.loc['Đã NV'], dr.loc['Đạt PV'])]
            st.dataframe(dd.style.format('{:,.0f}', subset=pd.IndexSlice[DEPT_METRICS, :]).format('{:.1f}%', na_rep='–', subset=pd.IndexSlice[['% Tham gia PV / CV', '% Đạt PV / Tham gia PV', '% Đã NV / Đạt PV'], :]), width='stretch')
            st.caption('Chỉ tính các (chi nhánh, vị trí) có trong danh sách ở sheet BC-2026 của file TUYỂN DỤNG - DATA, như báo cáo.')
        else: st.caption('Không đọc được danh sách vị trí (sheet BC-2026) trong file DATA.')
        st.subheader('5d. Người hết thử việc trong tháng (còn làm việc)')
        dtv = R['dat_thu_viec']; q_ = st.columns(len(dtv)); 
        for c_, (k_, v_) in zip(q_, dtv.items()): c_.metric(k_.title() if k_ != 'TỔNG' else 'Tổng', int(v_))
        st.caption('Theo ngày cuối thử việc trong Master, trạng thái "Đang làm việc".')

        st.markdown('---'); st.subheader('6. Người nhận việc có giữ được không?')
        rt = R['retention']
        b = st.columns(4); b[0].metric('Nhận việc trong tháng', rt['n']); b[1].metric('Đã nghỉ (tính đến nay)', rt['nghi_all'], fmt_pct(pct(rt['nghi_all'], rt['n'])), delta_color='off')
        b[2].metric('Nghỉ trong 30 ngày', rt['nghi30'], fmt_pct(pct(rt['nghi30'], rt['n'])), delta_color='off'); b[3].metric('Nghỉ trong 60 ngày', rt['nghi60'], fmt_pct(pct(rt['nghi60'], rt['n'])), delta_color='off')
        if not rt['du_30']: st.warning(f"Chưa đủ 30 ngày sau cuối tháng (dữ liệu đến {R['cutoff']:%d/%m/%Y}) → tỷ lệ nghỉ trong 30 ngày còn thấp hơn thực tế.")
        show(R['retention_src'].set_index('NGUON')[['NV', 'NGHI', 'NGHI30', '%Đã nghỉ', '%Nghỉ ≤30 ngày']] if len(R['retention_src']) else R['retention_src'])
        st.caption('Một người được coi là đã nghỉ khi có trong danh sách nghỉ việc (Kế toán/Master) với cùng họ tên và ngày nhận việc lệch ≤7 ngày. Người chưa nhập vào các file nghỉ việc sẽ bị tính là còn làm.')

        st.markdown('---'); st.subheader('7. Nghỉ việc')
        n = R['nghi_total']
        d_ = st.columns(4); d_[0].metric('Nghỉ việc', n); d_[1].metric('Nghỉ thử việc (<30 ngày)', R['nghi_tv'], fmt_pct(pct(R['nghi_tv'], n)), delta_color='off')
        d_[2].metric('Nghỉ ≤10 ngày', R['nghi_early10'], fmt_pct(pct(R['nghi_early10'], n)), delta_color='off'); d_[3].metric('Vào và nghỉ trong cùng tháng', R['nghi_moi'], fmt_pct(pct(R['nghi_moi'], n)), delta_color='off')
        if n:
            l, r_ = st.columns(2)
            with l: st.markdown('**Thời gian làm việc trước khi nghỉ**'); show(R['tenure']); st.bar_chart(R['tenure']['So_nguoi'])
            with r_: st.markdown('**Theo chi nhánh / bộ phận**'); show(R['nghi_unit'])
            l, r_ = st.columns(2)
            with l: st.markdown('**Theo chức vụ (top 15)**'); show(R['nghi_pos'].head(15))
            with r_: st.markdown('**Theo nguồn của người nghỉ**'); show(R['nghi_src'])
            st.markdown('**Nhóm lý do nghỉ**'); show(R['nghi_reason'])
            st.caption('Nhóm lý do do người làm báo cáo phân loại tay (12 nhóm, cộng 71 trong khi tổng nghỉ việc ghi 70 – đúng như báo cáo).' if R['reason_manual'] else 'Nhóm lý do được gán tự động theo từ khóa trong chữ Admin/Master nhập, chỉ để tham khảo.')
            if len(R['nghi_excluded']): st.warning(f"{len(R['nghi_excluded'])} người có ngày nghỉ trong tháng ở Kế toán/Master nhưng báo cáo không liệt kê nên không tính: " + '; '.join(R['nghi_excluded'].TEN) + '. Cần xác minh.')
        else: st.caption('Không có ai nghỉ việc trong tháng này theo Kế toán/Master.')

        st.markdown('---'); st.subheader('8. Chi phí tuyển dụng')
        e = st.columns(5); e[0].metric('Facebook Ads', fmt_vnd(c['fb'])); e[1].metric('Nhập tay (CareerViet/TopCV/khác)', fmt_vnd(sum(c['manual'].values()))); e[2].metric('Tổng chi phí', fmt_vnd(c['total'])); e[3].metric('Chi phí / CV', fmt_vnd(c['per_cv'])); e[4].metric('Chi phí / Pass', fmt_vnd(c['per_pass']))
        if c['ads_days'] == 0: st.caption('File Facebook Ads không có số liệu của tháng này (hoặc chưa tải lên).')
        else:
            st.caption(f"Facebook Ads có {c['ads_days']}/{c['days_in_month']} ngày trong tháng; {c['msg']:,.0f} lượt nhắn tin; chi phí/lượt nhắn tin {fmt_vnd(c['cpm'])}.")
            ad = R['ads_daily'].set_index('ngay'); st.bar_chart(ad['spend'])
        if c['total'] == 0: st.caption('Nhập chi phí CareerViet/TopCV ở thanh bên để tính chi phí trên mỗi CV / người nhận việc.')

        st.markdown('---'); st.subheader('9. Độ đầy đủ dữ liệu giữa các file')
        cv_ = R['coverage']
        cov = pd.DataFrame({'Chỉ tiêu': ['Người nhận việc (DATA tuyển dụng)', '  – có trong Master Data', '  – có trong Kế toán (NHẬN VIỆC, chi nhánh)', 'Số dòng nhận việc tháng trong Master', 'Số dòng nhận việc tháng trong Kế toán', 'Người nghỉ việc (hợp 3 nguồn)', '  – có trong Master', '  – có trong CHECK LIST', '  – có trong sheet NGHỈ VIỆC'],
                            'Số lượng': [cv_['nv_data'], cv_['nv_in_master'], cv_['nv_in_kt'], cv_['master_nv_month'], cv_['kt_nv_month'], cv_['nghi_n'], cv_['nghi_in_master'], cv_['nghi_in_ck'], cv_['nghi_in_kq']]})
        cov['%'] = [np.nan, pct(cv_['nv_in_master'], cv_['nv_data']), pct(cv_['nv_in_kt'], cv_['nv_data']), np.nan, np.nan, np.nan, pct(cv_['nghi_in_master'], cv_['nghi_n']), pct(cv_['nghi_in_ck'], cv_['nghi_n']), pct(cv_['nghi_in_kq'], cv_['nghi_n'])]
        show(cov.set_index('Chỉ tiêu'))
        dq = raw['dq']; msgs = []
        if R['dup_cv']: msgs.append(f"{R['dup_cv']} dòng CV trùng (cùng tên, vị trí, ngày liên lạc, nguồn) – vẫn được đếm như báo cáo.")
        txtn = dq['text_NGAYPV'] + dq['text_NGAYNV'] + dq['text_NGAYLL'] + dq['text_NGAYNGHI']
        if txtn: msgs.append(f"{txtn} ô ngày trong DATA nhập dạng chữ (vd \"03/04/2026\"); app đọc theo ngày/tháng/năm, còn công thức Excel của báo cáo bỏ qua các ô này.")
        if dq['master_nghi_khong_ngay']: msgs.append(f"Master có {dq['master_nghi_khong_ngay']} người trạng thái đã nghỉ nhưng không có ngày nghỉ hợp lệ nên không xếp được vào tháng nào.")
        if dq['data_thieu_ngay_lien_lac']: msgs.append(f"DATA có {dq['data_thieu_ngay_lien_lac']} dòng thiếu ngày liên lạc nên không được tính vào CV tháng nào.")
        if cv_['nghi_lech_ngay']: msgs.append(f"{cv_['nghi_lech_ngay']} người nghỉ có ngày nghỉ khác nhau giữa các file (trong đó {cv_['nghi_lech_thang']} người lệch sang tháng khác ở file kia). App lấy ngày theo thứ tự ưu tiên Master → CHECK LIST → sheet NGHỈ VIỆC của Kế toán.")
        for t_ in msgs: st.caption('⚠️ ' + t_)
        st.markdown('---'); st.subheader('10. Phân tích chuyên sâu')
        st.caption('Tính từ dòng raw của tháng (không gồm điều chỉnh khớp báo cáo) nên có thể lệch vài đơn vị so với các bảng ở trên khi đang bật "Khớp báo cáo".')
        t1, t2, t3, t4, t5, t6 = st.tabs(['Sàng lọc CV', 'Thời gian tuyển', 'Người phỏng vấn', 'Nguồn × chi nhánh / vị trí', 'Lương đề nghị', 'Giữ chân chi tiết'])
        with t1:
            if len(Dp['screen']):
                st.markdown('**Kết quả đánh giá CV và đi tiếp tới PV / Pass / nhận việc (đợt CV của tháng)**'); show(Dp['screen'])
                st.markdown('**Theo nguồn**'); show(Dp['screen_src'][['CV', 'DA_DANH_GIA', 'PHU_HOP', 'LUU', 'KHONG_PHU_HOP', '% Phù hợp', '% Lưu', '% Không phù hợp', '% Phù hợp đi tới PV', '% Phù hợp nhận việc']])
                st.markdown('**Theo HR**'); show(Dp['screen_hr'][['CV', 'DA_DANH_GIA', 'PHU_HOP', 'LUU', 'KHONG_PHU_HOP', '% Phù hợp', '% Lưu', '% Không phù hợp', '% Phù hợp đi tới PV', '% Phù hợp nhận việc']])
                st.caption('Dựa vào cột ĐÁNH GIÁ trong DATA (Phù hợp / Lưu / Không phù hợp). CV chưa đánh giá được tách riêng.')
            else: st.caption('Không có CV trong tháng.')
        with t2:
            lt = Dp['lead']
            if lt['n']:
                st.markdown(f"**Thời gian tuyển của {lt['n']} người nhận việc trong tháng (ngày)**"); show(lt['stats'])
                st.caption('"Ngày sai thứ tự" = ngày PV sớm hơn ngày liên lạc (nhập liệu sai); các dòng này bị loại khỏi trung bình/trung vị.')
                l, r_ = st.columns(2)
                with l: st.markdown('**Phân bố: liên lạc → nhận việc**'); show(lt['dist']); st.bar_chart(lt['dist']['So_nguoi'])
                with r_: st.markdown('**Theo nguồn**'); show(lt['by_source'])
                st.markdown('**Theo vị trí (top 15)**'); show(lt['by_pos'])
            else: st.caption('Không có người nhận việc trong tháng.')
        with t3:
            if len(Dp['interv']):
                show(Dp['interv']); st.caption('Gom các cách viết tên khác nhau của cùng một người (A Pháp / a Pháp / Pháp) – chỉ mang tính tham khảo. "Nhận việc / Pass" tính trên những người đã Pass trong tháng.')
            else: st.caption('Không có buổi phỏng vấn trong tháng.')
        with t4:
            for ttl, key in (('Nguồn × chi nhánh/bộ phận', 'x_src_unit'), ('Nguồn × vị trí', 'x_src_pos')):
                xm = Dp[key]
                if xm is None: st.caption('Không có dữ liệu.'); continue
                st.markdown(f'**{ttl}** (đợt CV của tháng)')
                sel_ = st.radio('Hiển thị', ['% nhận việc / CV', 'Số CV', 'Số nhận việc'], horizontal=True, key=f'{K}_{key}')
                if sel_.startswith('%'): st.dataframe(xm['rate'].style.format('{:.1f}%', na_rep='–'), width='stretch')
                else: show(xm['cv'] if sel_ == 'Số CV' else xm['nv'])
            st.caption('Ô trống (–) = không có CV. Ô có ít CV cho tỷ lệ dao động mạnh, xem cùng bảng "Số CV".')
        with t5:
            sal = Dp['salary']
            if sal.get('n'):
                st.markdown(f"Có {sal['n']} / {sal['cv']} CV ghi mức lương đề nghị ({sal['cover']:.1f}%). Trung vị {sal['overall']['median']:.1f} triệu, trung bình {sal['overall']['mean']:.1f} triệu.")
                show(sal['table']); st.caption('Chỉ những CV có ghi lương mới được tính (thường là đã chốt/đề xuất lương) nên "% nhận việc" ở đây cao hơn mặt bằng chung.')
            else: st.caption('Không có CV nào ghi mức lương trong tháng.')
        with t6:
            rd = Dp['ret']; st.markdown('**Đường cong giữ chân của người nhận việc trong tháng**'); show(rd['curve'])
            st.caption(f"Chỉ tính người đã đủ số ngày tính tới {R['cutoff']:%d/%m/%Y}. Người chưa có trong file nghỉ việc được coi là còn làm.")
            for ttl, key in (('Nghỉ trong 30 ngày theo nguồn', 'by_source'), ('theo chi nhánh/bộ phận', 'by_unit'), ('theo vị trí', 'by_pos'), ('theo HR', 'by_hr')):
                st.markdown(f'**{ttl}**'); show(rd[key])
        with st.expander('Danh sách chi tiết người nhận việc'): show(R['nv_list'].drop(columns=['k']), hide_index=True)
        with st.expander('Danh sách chi tiết người nghỉ việc'): show(R['nghi_list'].drop(columns=['k']) if len(R['nghi_list']) else R['nghi_list'], hide_index=True)



    # ---------------------------------------------------------------- THEO TUẦN
    def get_ktv(R):
        key = ('ktv', id(raw), R['start'], R['end'], R['adjusted'])
        if key not in st.session_state['res']: st.session_state['res'][key] = ktv_tvv(R)
        return st.session_state['res'][key]
    with tabs[2]:
        st.subheader('Theo tuần')
        st.caption('Tuần tính từ thứ Hai đến Chủ nhật, đánh số theo năm (giống "TUẦN 40" ở file Kế toán). Tuần đầu/cuối có dấu * là tuần bị cắt theo ranh giới tháng. Tổng các tuần luôn bằng số cả tháng. Số liệu tính từ raw (không gồm điều chỉnh khớp báo cáo).')
        opts = []
        for R_ in results:
            m_ = R_['month']; Wk = get_weekly(m_); tt = Wk['table']; st.markdown(f'### {MONTHS[m_]}/{year}')
            show(tt[['Từ', 'Đến', 'CV', 'PV', 'PASS', 'KHONG_DEN_PV', 'NV', 'KHONG_NV', 'NGHI', 'NGHI_TV', 'NGHI_SOM', '% PV/CV', '% Pass/PV', '% NV/Pass', '% NV/CV', '% Nghỉ/Nhận việc', '% Nghỉ thử việc/Nghỉ']])
            st.bar_chart(tt.drop(index='CẢ THÁNG')[['CV', 'PV', 'PASS', 'NV', 'NGHI']])
            with st.expander(f'Theo HR × tuần – {MONTHS[m_]}'):
                for k_, nm_ in (('CV', 'Số CV'), ('PV', 'Tham gia PV'), ('PASS', 'Đạt PV'), ('NV', 'Nhận việc')): st.markdown(f'**{nm_}**'); show(Wk['hr'][k_])
            with st.expander(f'Theo nguồn × tuần – {MONTHS[m_]}'):
                for k_, nm_ in (('CV', 'Số CV'), ('PASS', 'Đạt PV'), ('NV', 'Nhận việc')): st.markdown(f'**{nm_}**'); show(Wk['src'][k_])
            with st.expander(f'Theo chi nhánh / bộ phận × tuần – {MONTHS[m_]}'):
                for k_, nm_ in (('CV', 'Số CV'), ('NV', 'Nhận việc'), ('NGHI', 'Nghỉ việc')): st.markdown(f'**{nm_}**'); show(Wk['unit'][k_])
            with st.expander(f'KTV / TVV theo chi nhánh và thời gian làm việc trước khi nghỉ (cả tháng) – {MONTHS[m_]}'):
                Kt = get_ktv(R_); show(Kt['cn'])
                if len(Kt['tenure']): st.markdown('**Thời gian làm việc trước khi nghỉ**'); show(Kt['tenure'])
                if len(Kt['khac_nv']) or len(Kt['khac_ng']): st.caption('Nhóm "Khác" gồm các vị trí không phải KTV/TVV: ' + ', '.join(f'{k_} ({v_})' for k_, v_ in list(Kt['khac_nv'].items())[:8] + list(Kt['khac_ng'].items())[:8]))
            opts += [(m_, w_) for w_ in Wk['weeks']]
        st.markdown('---'); st.subheader('Xem chi tiết một tuần')
        labels_ = [f"{MONTHS[m_]} – {w_['label']}" for m_, w_ in opts]
        pick_ = st.selectbox('Chọn tuần', labels_, key='wk_pick')
        m_, w_ = opts[labels_.index(pick_)]
        render_period(get_period(w_['start'], w_['end'], f"{w_['label']} – {MONTHS[m_]}/{year}", m_), f"w{m_}_{w_['no']}")

    # ---------------------------------------------------------------- KIỂM TRA DỮ LIỆU
    with tabs[3]:
        st.subheader('Kiểm tra dữ liệu giữa DATA – Kế toán – Master')
        st.caption('App ghép người giữa các file theo họ tên + ngày nhận việc/nghỉ. Nếu ai đó nhập sai tên thì người đó bị coi là "chưa nhập" ở file kia, hoặc bị đếm thành 2 người (khi nghỉ việc). Sheet này liệt kê các trường hợp nghi sai tên (tên giống nhau từ 85%, cùng ngày nhận việc), người chưa nhập, ngày/chi nhánh lệch và ngày vô lý để bạn sửa trong file gốc. App KHÔNG tự sửa hay tự gộp tên.')
        frames = []
        for R_ in results:
            c_ = get_checks(R_).copy(); c_.insert(0, 'Tháng', f"T{R_['month']:02d}"); frames.append(c_)
        allc = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if not len(allc): st.success('Không phát hiện vấn đề nào trong các tháng đã chọn.')
        else:
            q = st.columns(4); q[0].metric('Tổng số vấn đề', len(allc)); [q[i + 1].metric(sv_, int((allc['Mức độ'] == sv_).sum())) for i, sv_ in enumerate(['Cao', 'Trung bình', 'Thấp'])]
            f1, f2, f3 = st.columns(3)
            sev_ = f1.multiselect('Mức độ', ['Cao', 'Trung bình', 'Thấp'], default=['Cao', 'Trung bình', 'Thấp'], key='chk_sev')
            typ_ = f2.multiselect('Loại vấn đề', sorted(allc['Loại'].unique()), default=[], key='chk_type', help='Để trống = tất cả')
            nm_q = f3.text_input('Tìm theo họ tên', key='chk_name')
            v_ = allc[allc['Mức độ'].isin(sev_)]
            if typ_: v_ = v_[v_['Loại'].isin(typ_)]
            if nm_q.strip(): v_ = v_[v_['Họ tên'].astype(str).map(norm).str.contains(norm(nm_q), regex=False)]
            st.markdown('**Tổng hợp theo loại**'); show(allc.groupby(['Mức độ', 'Loại']).size().rename('Số lượng').reset_index().assign(_o=lambda x: x['Mức độ'].map(SEV_ORDER)).sort_values(['_o', 'Số lượng'], ascending=[True, False]).drop(columns='_o'), hide_index=True)
            st.markdown(f'**Chi tiết ({len(v_)} dòng)**'); st.dataframe(v_, width='stretch', hide_index=True, column_config={'Chi tiết': st.column_config.TextColumn(width='large'), 'Gợi ý xử lý': st.column_config.TextColumn(width='medium')})
            st.caption('Cột "Dòng" là số dòng trong sheet của file gốc để bạn mở đúng chỗ sửa. Mức độ Cao = ảnh hưởng tới số liệu (đếm trùng/thiếu); Trung bình = lệch giữa các file; Thấp = nhập liệu chưa chuẩn.')

    # ---------------------------------------------------------------- KHOẢNG NGÀY TÙY CHỌN
    Rr = None
    if rng:
        with tabs[4]:
            Rr = get_period(rng[0], rng[1], f"Từ {rng[0]:%d/%m/%Y} đến {rng[1]:%d/%m/%Y}", rng[1].month)
            st.caption('Khoảng ngày tùy chọn (như dashboard theo dõi): mỗi chỉ tiêu tính theo ngày của chính bước đó trong khoảng đã chọn. Số liệu từ raw, không gồm điều chỉnh khớp báo cáo.')
            render_period(Rr, 'rng')

    for tab, R in zip(tabs[NF:], results):
        with tab: render_period(R, f"m{R['month']}")
    # ---- xuất Excel (đặt ở thanh bên; dựng lại khi lựa chọn thay đổi)
    deeps = {R['month']: get_deep(R) for R in results}
    wk_x = {R['month']: dict(week=get_weekly(R['month']), ktv=get_ktv(R)) for R in results}; ck_x = {R['month']: get_checks(R) for R in results}; cu_x = (Rr, get_deep(Rr)) if Rr is not None else None
    sig = repr((id(raw), year, tuple(sel), use_adj, rng, tuple((m_, tuple(sorted(manual[m_].items()))) for m_ in sel), tuple((m_, tuple(map(tuple, bt_store[m_].fillna(-1).round(4).values.tolist()))) for m_ in sorted(bt_store))))
    if st.session_state.get('xl_sig') != sig:
        st.session_state['xl_bytes'] = export_excel(results, deeps, bt_store, get_trend(), dict(files=names), weekly=wk_x, checks=ck_x, custom=cu_x); st.session_state['xl_sig'] = sig
    xl_slot.download_button('⬇️ Tải báo cáo Excel (đầy đủ)', st.session_state['xl_bytes'], file_name='Bao_cao_tuyen_dung_' + '_'.join(f'T{m_:02d}' for m_ in sel) + f'_{year}.xlsx', key='dl_all',
                            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

if __name__ == '__main__':
    main()
