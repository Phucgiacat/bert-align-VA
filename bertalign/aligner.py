import numpy as np
import csv
import json
import os

import bertalign
from bertalign.corelib import *
from bertalign.utils import *

class Bertalign:
    def __init__(self,
                 src,
                 tgt,
                 max_align=5,
                 top_k=3,
                 win=5,
                 skip=-0.1,
                 margin=True,
                 len_penalty=True,
                 is_split=False,
                 vi_word_segmentation=True,
               ):
        
        self.max_align = max_align
        self.top_k = top_k
        self.win = win
        self.skip = skip
        self.margin = margin
        self.len_penalty = len_penalty
        
        src = clean_text(src)
        tgt = clean_text(tgt)
        src_lang = detect_lang(src)
        tgt_lang = detect_lang(tgt)
        
        if is_split:
            src_sents = src.splitlines()
            tgt_sents = tgt.splitlines()
        else:
            src_sents = split_sents(src, src_lang)
            tgt_sents = split_sents(tgt, tgt_lang)
 
        src_num = len(src_sents)
        tgt_num = len(tgt_sents)
        
        src_lang = LANG.ISO[src_lang]
        tgt_lang = LANG.ISO[tgt_lang]
        
        print("Source language: {}, Number of sentences: {}".format(src_lang, src_num))
        print("Target language: {}, Number of sentences: {}".format(tgt_lang, tgt_num))

        model = bertalign.get_model()
        print("Embedding source and target text using {} ...".format(model.model_name))
        
        # Word segmentation using pyvi for Vietnamese
        if vi_word_segmentation:
            try:
                from pyvi import ViTokenizer
            except ImportError:
                print("Warning: pyvi is not installed. Run 'pip install pyvi' to use Vietnamese word segmentation. Skipping segmentation.")
                src_sents_embed = src_sents
                tgt_sents_embed = tgt_sents
            else:
                src_sents_embed = [ViTokenizer.tokenize(s) if src_lang == 'Vietnamese' else s for s in src_sents]
                tgt_sents_embed = [ViTokenizer.tokenize(s) if tgt_lang == 'Vietnamese' else s for s in tgt_sents]
        else:
            src_sents_embed = src_sents
            tgt_sents_embed = tgt_sents

        model = bertalign.get_model()
        src_vecs, src_lens = model.transform(src_sents_embed, max_align - 1)
        tgt_vecs, tgt_lens = model.transform(tgt_sents_embed, max_align - 1)

        char_ratio = np.sum(src_lens[0,]) / np.sum(tgt_lens[0,])

        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.src_sents = src_sents
        self.tgt_sents = tgt_sents
        self.src_num = src_num
        self.tgt_num = tgt_num
        self.src_lens = src_lens
        self.tgt_lens = tgt_lens
        self.char_ratio = char_ratio
        self.src_vecs = src_vecs
        self.tgt_vecs = tgt_vecs
        
    def align_sents(self):

        print("Performing first-step alignment ...")
        D, I = find_top_k_sents(self.src_vecs[0,:], self.tgt_vecs[0,:], k=self.top_k)
        first_alignment_types = get_alignment_types(2) # 0-1, 1-0, 1-1
        first_w, first_path = find_first_search_path(self.src_num, self.tgt_num)
        first_pointers = first_pass_align(self.src_num, self.tgt_num, first_w, first_path, first_alignment_types, D, I)
        first_alignment = first_back_track(self.src_num, self.tgt_num, first_pointers, first_path, first_alignment_types)
        
        print("Performing second-step alignment ...")
        second_alignment_types = get_alignment_types(self.max_align)
        second_w, second_path = find_second_search_path(first_alignment, self.win, self.src_num, self.tgt_num)
        second_pointers = second_pass_align(self.src_vecs, self.tgt_vecs, self.src_lens, self.tgt_lens,
                                            second_w, second_path, second_alignment_types,
                                            self.char_ratio, self.skip, margin=self.margin, len_penalty=self.len_penalty)
        second_alignment = second_back_track(self.src_num, self.tgt_num, second_pointers, second_path, second_alignment_types)
        
        print("Finished! Successfully aligning {} {} sentences to {} {} sentences\n".format(self.src_num, self.src_lang, self.tgt_num, self.tgt_lang))
        self.result = second_alignment
        self._post_process_alignment()
    
    def _is_noise(self, text):
        import re
        text = text.strip()
        if not text:
            return True
        # Check if text is only punctuation, spaces, or very short and meaningless
        if re.match(r'^[\s\.\,\;\:\!\?\-\"\'\…\“\”]+$', text):
            return True
        return False
        
    def _merge_beads(self, bead1, bead2):
        # Mở rộng index của src và tgt
        src_idx = list(bead1[0]) + list(bead2[0])
        tgt_idx = list(bead1[1]) + list(bead2[1])
        # Loại bỏ trùng lặp và sắp xếp
        src_idx = sorted(list(set(src_idx)))
        tgt_idx = sorted(list(set(tgt_idx)))
        return (src_idx, tgt_idx)

    def _post_process_alignment(self):
        """Clean up alignment artifacts: empty pairs, noise, orphans."""
        cleaned = []
        for i, bead in enumerate(self.result):
            src = self._get_line(bead[0], self.src_sents).strip()
            tgt = self._get_line(bead[1], self.tgt_sents).strip()
            # Skip fully empty pairs
            if not src and not tgt:
                continue
            # Merge noise into previous pair
            if self._is_noise(src) or self._is_noise(tgt):
                if cleaned:
                    cleaned[-1] = self._merge_beads(cleaned[-1], bead)
                    continue
            cleaned.append(bead)
        self.result = cleaned
    
    def print_sents(self):
        for bead in (self.result):
            src_line = self._get_line(bead[0], self.src_sents)
            tgt_line = self._get_line(bead[1], self.tgt_sents)
            print(src_line + "\n" + tgt_line + "\n")

    def get_alignments(self):
        """
        Trả về danh sách các cặp câu đã align dưới dạng list of dict.
        Mỗi dict chứa:
            - pair_id    : thứ tự cặp (bắt đầu từ 1)
            - align_type : loại alignment, ví dụ '1-1', '1-2', '2-1', '0-1'
            - src_idx    : list chỉ số câu nguồn (0-indexed)
            - tgt_idx    : list chỉ số câu đích (0-indexed)
            - src        : câu nguồn (nối bằng dấu cách nếu nhiều câu)
            - tgt        : câu đích (nối bằng dấu cách nếu nhiều câu)
        """
        records = []
        for i, bead in enumerate(self.result):
            src_idx = list(bead[0])
            tgt_idx = list(bead[1])
            src_line = self._get_line(bead[0], self.src_sents)
            tgt_line = self._get_line(bead[1], self.tgt_sents)
            records.append({
                'pair_id': i + 1,
                'align_type': "{}-{}".format(len(src_idx), len(tgt_idx)),
                'src_idx': src_idx,
                'tgt_idx': tgt_idx,
                'src': src_line,
                'tgt': tgt_line,
            })
        return records

    def save_sents(self, output_path, format='csv', include_metadata=False):
        """
        Xuất kết quả alignment ra file.

        Tham số:
            output_path (str)       : Đường dẫn file đầu ra.
            format (str)            : Định dạng xuất. Hỗ trợ:
                                        'csv'  - Comma-Separated Values (.csv)
                                        'tsv'  - Tab-Separated Values (.tsv) — chuẩn NLP
                                        'json' - JSON với đầy đủ metadata (.json)
                                        'txt'  - Plain text song ngữ, tab-separated (.txt)
            include_metadata (bool) : Nếu True, thêm các cột pair_id, align_type,
                                      src_idx, tgt_idx vào CSV/TSV.
                                      Không ảnh hưởng đến JSON (luôn có metadata) và TXT.

        Ví dụ sử dụng:
            aligner.save_sents('output.csv')
            aligner.save_sents('output.tsv', format='tsv')
            aligner.save_sents('output.json', format='json')
            aligner.save_sents('output.txt', format='txt')
            aligner.save_sents('output_meta.csv', include_metadata=True)
        """
        fmt = format.lower().strip()
        records = self.get_alignments()

        if fmt in ('csv', 'tsv'):
            sep = '\t' if fmt == 'tsv' else ','
            with open(output_path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f, delimiter=sep)
                if include_metadata:
                    writer.writerow(['pair_id', 'align_type', 'src_idx', 'tgt_idx',
                                     self.src_lang, self.tgt_lang])
                    for r in records:
                        writer.writerow([
                            r['pair_id'],
                            r['align_type'],
                            ';'.join(str(i) for i in r['src_idx']),
                            ';'.join(str(i) for i in r['tgt_idx']),
                            r['src'],
                            r['tgt'],
                        ])
                else:
                    writer.writerow([self.src_lang, self.tgt_lang])
                    for r in records:
                        writer.writerow([r['src'], r['tgt']])

        elif fmt == 'json':
            output = {
                'metadata': {
                    'src_lang': self.src_lang,
                    'tgt_lang': self.tgt_lang,
                    'src_sentences_total': self.src_num,
                    'tgt_sentences_total': self.tgt_num,
                    'alignment_pairs': len(records),
                },
                'alignments': records,
            }
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output, f, ensure_ascii=False, indent=2)

        elif fmt == 'txt':
            with open(output_path, 'w', encoding='utf-8') as f:
                for r in records:
                    if r['src'] or r['tgt']:
                        f.write(r['src'] + '\t' + r['tgt'] + '\n')

        else:
            raise ValueError(
                "Định dạng '{}' không được hỗ trợ. Chọn một trong: csv, tsv, json, txt".format(fmt)
            )

        print("Đã lưu {} cặp câu align sang '{}' (định dạng {})".format(
            len(records), output_path, fmt.upper()
        ))

    @staticmethod
    def _get_line(bead, lines):
        line = ''
        if len(bead) > 0:
            line = ' '.join(lines[bead[0]:bead[-1]+1])
        return line
