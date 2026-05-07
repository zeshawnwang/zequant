"""Unit tests for FeeCalculator."""
import sys
sys.path.insert(0, '.')

from core.fee import FeeCalculator


def test_fee_buy():
    fee_calc = FeeCalculator()
    cost = fee_calc.calc_buy('000001', 10.0, 1000)
    # 买入: 佣金 + 过户费 + 滑点
    assert cost.commission >= 3.0  # 最低佣金5元，实际按比例10000*0.0003=3
    assert cost.stamp_tax == 0  # 买入无印花税
    print(f"买入费用: {cost}")


def test_fee_sell():
    fee_calc = FeeCalculator()
    cost = fee_calc.calc_sell('000001', 10.0, 1000)
    # 卖出: 佣金 + 过户费 + 印花税 + 滑点
    assert cost.stamp_tax > 0  # 卖出收印花税
    print(f"卖出费用: {cost}")


def test_fee_shanghai():
    fee_calc = FeeCalculator()
    buy_sh = fee_calc.calc_buy('600000', 10.0, 1000)
    buy_sz = fee_calc.calc_buy('000001', 10.0, 1000)
    # 沪市有过户费，深市无
    assert buy_sh.transfer_fee > buy_sz.transfer_fee
    print(f"沪市买入过户费: {buy_sh.transfer_fee:.2f}")
    print(f"深市买入过户费: {buy_sz.transfer_fee:.2f}")


if __name__ == "__main__":
    test_fee_buy()
    test_fee_sell()
    test_fee_shanghai()
    print("费用测试通过!")
