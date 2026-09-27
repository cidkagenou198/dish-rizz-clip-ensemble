import re
import unicodedata
import pandas as pd
from pathlib import Path

def normalize_text(text: str) -> str:
    """Normalize text: lowercase, remove non-printable/replacement chars, normalize accents."""
    if not isinstance(text, str):
        return ""
    # Normalize unicode (e.g. accented characters)
    text = unicodedata.normalize('NFKD', text)
    # Replace common corruptions or replacement characters
    text = text.replace('\ufffd', 'e').replace('', 'e')
    return text.lower().strip()

class QueryMapper:
    def __init__(self, metadata_path: str = "category_metadata.csv"):
        self.metadata_df = pd.read_csv(metadata_path)
        self.cat_to_id = {}
        self.id_to_name = {}
        self.id_to_idx = {}
        self.idx_to_id = {}
        self.name_to_id = {}
        
        # Populate category indices
        for idx, row in self.metadata_df.iterrows():
            cid = row['category_id']
            cname = row['category_name'].lower().strip()
            self.cat_to_id[cname] = cid
            self.id_to_name[cid] = cname
            self.id_to_idx[cid] = idx
            self.idx_to_id[idx] = cid
            
            # Map clean name
            norm_cname = normalize_text(cname)
            self.name_to_id[norm_cname] = cid
            
            # Map alternative names
            if pd.notna(row['alternative_names']):
                alts = re.split(r'[,|]', str(row['alternative_names']))
                for alt in alts:
                    alt_norm = normalize_text(alt)
                    if alt_norm:
                        self.name_to_id[alt_norm] = cid

        # Sort patterns by length descending so longer phrases match first
        self.sorted_patterns = sorted(self.name_to_id.keys(), key=lambda x: len(x), reverse=True)

    def map_query(self, description: str) -> str:
        """Map query string to category_id."""
        norm_desc = normalize_text(description)
        for pattern in self.sorted_patterns:
            regex = r'\b' + re.escape(pattern) + r'\b'
            if re.search(regex, norm_desc):
                return self.name_to_id[pattern]
        
        # Fallback substring match
        for pattern in self.sorted_patterns:
            if pattern in norm_desc:
                return self.name_to_id[pattern]
                
        raise ValueError(f"Could not map query description to any category: {description}")

    def map_dataframe(self, df: pd.DataFrame, desc_col: str = "description") -> pd.DataFrame:
        """Add category_id and category_idx columns to queries dataframe."""
        df = df.copy()
        df['category_id'] = df[desc_col].apply(self.map_query)
        df['category_idx'] = df['category_id'].apply(lambda x: self.id_to_idx[x])
        df['category_name'] = df['category_id'].apply(lambda x: self.id_to_name[x])
        return df

if __name__ == '__main__':
    qm = QueryMapper()
    val_q = pd.read_csv('validation_queries.csv')
    val_mapped = qm.map_dataframe(val_q)
    val_gt = pd.read_csv('validation_ground_truth.csv')
    acc = (val_mapped['category_id'] == val_gt['category_id']).mean()
    print(f"Validation mapping accuracy: {acc*100:.2f}%")
    
    test_q = pd.read_csv('test_queries.csv')
    test_mapped = qm.map_dataframe(test_q)
    print(f"Test mapping successfully mapped all {len(test_mapped)} queries!")
    print(f"Unique categories in test: {test_mapped['category_id'].nunique()}/101")
