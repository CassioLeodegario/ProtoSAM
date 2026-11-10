
from types import MethodType

class PrototiposVaziosNoAlinhamento(Exception):
    pass

def instalar(modelo):
    original_align = modelo.alignLoss
    original_prediction = modelo.cls_unit.get_prediction_from_prototypes
    estado = {"ativo": False, "ultimo": None}

    def prediction(self, prototypes, query, mode, *args, **kwargs):
        if estado["ativo"] and prototypes.shape[0] == 0:
            raise PrototiposVaziosNoAlinhamento(str(mode))
        return original_prediction(
            prototypes, query, mode, *args, **kwargs
        )

    def align(self, qry_fts, pred, supp_fts, fore_mask, back_mask):
        estado["ativo"] = True
        try:
            return original_align(
                qry_fts, pred, supp_fts, fore_mask, back_mask
            )
        except PrototiposVaziosNoAlinhamento as erro:
            estado["ultimo"] = {
                "motivo": "zero prototipos no alinhamento",
                "modo": str(erro),
                "acao": "alinhamento zero; CE preservada",
            }
            return qry_fts.sum() * 0.0
        finally:
            estado["ativo"] = False

    modelo.alignLoss = MethodType(align, modelo)
    modelo.cls_unit.get_prediction_from_prototypes = MethodType(
        prediction, modelo.cls_unit
    )
    return estado
