//! `account_hash` of the Accounts Delta Root SIMD.

use {solana_hash::Hash, solana_pubkey::Pubkey};

/// `account_hash` of an account:
///
/// ```text
/// 32 zero bytes                                           if lamports == 0
/// blake3_256(u64_le(lamports) || data || u8(executable) || owner || pubkey)  otherwise
/// ```
///
/// The input is the single-account input of the accounts lattice hash
/// (SIMD-0215), so the value equals the first 32 bytes of the account's
/// lattice hash, which is what validators commit. An address with zero
/// lamports has no account; its hash is zero.
pub fn account_hash(
    lamports: u64,
    data: &[u8],
    executable: bool,
    owner: &Pubkey,
    pubkey: &Pubkey,
) -> Hash {
    if lamports == 0 {
        return Hash::default();
    }
    let mut hasher = blake3::Hasher::new();
    hasher.update(&lamports.to_le_bytes());
    hasher.update(data);
    hasher.update(&[u8::from(executable)]);
    hasher.update(owner.as_ref());
    hasher.update(pubkey.as_ref());
    Hash::new_from_array(*hasher.finalize().as_bytes())
}

#[cfg(test)]
mod tests {
    use {
        super::*,
        solana_account::{AccountSharedData, ReadableAccount, WritableAccount},
        solana_accounts_db::accounts_db::AccountsDb,
    };

    /// The first 32 bytes of the lattice hash the runtime computes for an
    /// account (`AccountsDb::lt_hash_account`). The lattice hash elements are
    /// the BLAKE3 output bytes read as native-endian `u16`s.
    fn lt_hash_prefix(account: &AccountSharedData, pubkey: &Pubkey) -> Hash {
        let lt_hash = AccountsDb::lt_hash_account(account, pubkey).0;
        let bytes: Vec<u8> = lt_hash.0[..16]
            .iter()
            .flat_map(|element| element.to_ne_bytes())
            .collect();
        Hash::new_from_array(bytes.try_into().unwrap())
    }

    fn check(account: &AccountSharedData, pubkey: &Pubkey) {
        let ours = account_hash(
            account.lamports(),
            account.data(),
            account.executable(),
            account.owner(),
            pubkey,
        );
        assert_eq!(ours, lt_hash_prefix(account, pubkey));
        assert_eq!(
            ours,
            solana_runtime::bank::accounts_delta::account_hash(account, pubkey)
        );
    }

    #[test]
    fn test_account_hash_is_lattice_hash_prefix() {
        let owner = Pubkey::new_unique();
        for (lamports, data_len, executable) in [
            (1, 0, false),
            (42, 1, false),
            (1_000_000, 100, true),
            (u64::MAX, 10 * 1024, false),
        ] {
            let data = (0..data_len).map(|i| i as u8).collect();
            let mut account = AccountSharedData::new_with_data(lamports, data, &owner);
            account.set_executable(executable);
            check(&account, &Pubkey::new_unique());
        }
    }

    #[test]
    fn test_zero_lamport_account_hash() {
        let owner = Pubkey::new_unique();
        let pubkey = Pubkey::new_unique();
        assert_eq!(
            account_hash(0, &[1, 2, 3], true, &owner, &pubkey),
            Hash::default()
        );
        // The runtime's zero-lamport lattice hash is the identity (all zero).
        let account = AccountSharedData::new_with_data(0, vec![1, 2, 3], &owner);
        assert_eq!(lt_hash_prefix(&account, &pubkey), Hash::default());
        check(&account, &pubkey);
    }

    #[test]
    fn test_account_hash_binds_every_field() {
        let owner = Pubkey::new_unique();
        let pubkey = Pubkey::new_unique();
        let base = account_hash(10, &[1, 2], false, &owner, &pubkey);
        assert_ne!(base, account_hash(11, &[1, 2], false, &owner, &pubkey));
        assert_ne!(base, account_hash(10, &[1, 3], false, &owner, &pubkey));
        assert_ne!(base, account_hash(10, &[1, 2], true, &owner, &pubkey));
        assert_ne!(
            base,
            account_hash(10, &[1, 2], false, &Pubkey::new_unique(), &pubkey)
        );
        assert_ne!(
            base,
            account_hash(10, &[1, 2], false, &owner, &Pubkey::new_unique())
        );
    }
}
