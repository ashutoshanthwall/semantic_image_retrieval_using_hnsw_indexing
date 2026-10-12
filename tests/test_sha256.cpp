#include <gtest/gtest.h>

#include <string>

#include "algorithms/basic.hpp"
#include "persistence/sha256.hpp"

using namespace sir;

TEST(Sha256, NistVectors) {
    EXPECT_EQ(sha256Hex("", 0),
              "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    EXPECT_EQ(sha256Hex("abc", 3),
              "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    const std::string m = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
    EXPECT_EQ(sha256Hex(m.data(), m.size()),
              "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
    const std::string million(1000000, 'a');
    EXPECT_EQ(sha256Hex(million.data(), million.size()),
              "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
}

TEST(Sha256, IncrementalEqualsOneShot) {
    std::string s(1000, '\0');
    for (std::size_t i = 0; i < s.size(); ++i) s[i] = static_cast<char>(i * 7);
    Sha256 h;
    for (std::size_t off = 0; off < s.size(); off += 37)
        h.update(s.data() + off, minOf<std::size_t>(37, s.size() - off));
    const auto d = h.finish();
    EXPECT_EQ(toHex(d), sha256Hex(s.data(), s.size()));
}
