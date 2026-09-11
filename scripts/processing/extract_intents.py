"""ETL script to download Banking77 and extract its main columns and categories."""

from datasets import load_dataset

from intent_classification.config import INTERIM_DATA_DIR


GROUP_TO_INTENTS = {
    'Account': [
        'age_limit',
        'change_pin',
        'compromised_card',
        'country_support',
        'edit_personal_details',
        'lost_or_stolen_card',
        'lost_or_stolen_phone',
        'passcode_forgotten',
        'pin_blocked',
        'terminate_account',
        'unable_to_verify_identity',
        'verify_my_identity',
        'verify_source_of_funds',
        'why_verify_identity',
    ],
    'Cards': [
        'activate_my_card',
        'card_about_to_expire',
        'card_arrival',
        'card_delivery_estimate',
        'card_linking',
        'card_not_working',
        'card_swallowed',
        'contactless_not_working',
        'disposable_card_limits',
        'get_disposable_virtual_card',
        'get_physical_card',
        'getting_spare_card',
        'getting_virtual_card',
        'order_physical_card',
        'virtual_card_not_working',
        'visa_or_mastercard',
        'supported_cards_and_currencies',
    ],
    'Disputes': [
        'Refund_not_showing_up',
        'card_payment_not_recognised',
        'card_payment_wrong_exchange_rate',
        'cash_withdrawal_not_recognised',
        'direct_debit_payment_not_recognised',
        'extra_charge_on_statement',
        'request_refund',
        'reverted_card_payment?',
        'transaction_charged_twice',
        'wrong_amount_of_cash_received',
        'wrong_exchange_rate_for_cash_withdrawal',
    ],
    'Payments': [
        'atm_support',
        'card_acceptance',
        'card_payment_fee_charged',
        'cash_withdrawal_charge',
        'declined_card_payment',
        'declined_cash_withdrawal',
        'exchange_charge',
        'exchange_rate',
        'exchange_via_app',
        'fiat_currency_support',
        'pending_card_payment',
        'pending_cash_withdrawal',
    ],
    'Topups': [
        'apple_pay_or_google_pay',
        'automatic_top_up',
        'balance_not_updated_after_bank_transfer',
        'balance_not_updated_after_cheque_or_cash_deposit',
        'pending_top_up',
        'top_up_by_bank_transfer_charge',
        'top_up_by_card_charge',
        'top_up_by_cash_or_cheque',
        'top_up_failed',
        'top_up_limits',
        'top_up_reverted',
        'topping_up_by_card',
        'verify_top_up',
    ],
    'Transfers': [
        'beneficiary_not_allowed',
        'cancel_transfer',
        'declined_transfer',
        'failed_transfer',
        'pending_transfer',
        'receiving_money',
        'transfer_fee_charged',
        'transfer_into_account',
        'transfer_not_received_by_recipient',
        'transfer_timing',
    ],
}

INTENT_TO_GROUP = {
    intent: group for group, intents in GROUP_TO_INTENTS.items() for intent in intents
}


def extract_banking77_splits() -> None:
    """
    Download Banking77 from Hugging Face and extract the main columns and categories.

    Saves the text and label columns of each split to data/interim/.
    """
    print('Downloading dataset from Hugging Face...')
    dataset = load_dataset('mteb/banking77')

    for split_name in ['train', 'test']:
        df = dataset[split_name].to_pandas()

        df = df[['text', 'label_text']].rename(columns={'label_text': 'label'})
        df['label'] = df['label'].map(INTENT_TO_GROUP)
        print(f'- {split_name} split')
        print(f'  Rows: {len(df)}')
        print(f'  Classes: {df["label"].nunique()}')

        output_path = INTERIM_DATA_DIR / f'intents_{split_name}.csv'
        print(f'  Saving to {output_path}...')
        df.to_csv(output_path, index=False)

    print('-' * 40)


def main() -> None:
    """
    Extract the Banking77 splits and save the derived dataset to interim.

    Saves data/interim/intents_train.csv and data/interim/intents_test.csv.
    """
    extract_banking77_splits()
    print('Done!')


if __name__ == '__main__':
    main()
