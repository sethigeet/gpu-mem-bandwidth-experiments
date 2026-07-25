#include "chunked_hash_tree_rl.hpp"

#include <algorithm>
#include <climits>
#include <xxhash.h>

ChunkedHashTree_RL::ChunkedHashTree_RL(uint32_t chunk)
    : chunk_size(chunk), cache_valid(false), shared_chain_length(0)
{
    request_hashes.reserve(1024);
    missing_cache.reserve(1024);
    hash_to_waiting.reserve(4096);
    working_set.reserve(4096);
    hash_ref_counts.reserve(4096);
    shared_prefix_chain.reserve(256);
}

uint64_t ChunkedHashTree_RL::pack_key(uint32_t level, uint64_t hash)
{
    return (uint64_t(level) << 56) | (hash & 0x00FFFFFFFFFFFFFFULL);
}

void ChunkedHashTree_RL::invalidate_cache()
{
    cache_valid = false;
    cached_best_request.reset();
}

void ChunkedHashTree_RL::recompute_shared_chain()
{
    shared_prefix_chain.clear();
    shared_chain_length = 0;
    if (active_requests.empty()) return;

    uint32_t max_length = UINT32_MAX;
    for (auto request_id : active_requests) {
        auto request = request_hashes.find(request_id);
        if (request != request_hashes.end()) {
            max_length = std::min(max_length, (uint32_t)request->second.size());
        }
    }
    if (max_length == 0 || max_length == UINT32_MAX) return;

    for (uint32_t level = 0; level < max_length; ++level) {
        uint64_t common_hash = 0;
        bool first = true;
        bool all_match = true;
        for (auto request_id : active_requests) {
            auto request = request_hashes.find(request_id);
            if (
                request == request_hashes.end()
                || level >= request->second.size()) {
                all_match = false;
                break;
            }
            uint64_t hash = request->second[level];
            if (first) {
                common_hash = hash;
                first = false;
            } else if (hash != common_hash) {
                all_match = false;
                break;
            }
        }
        if (!all_match) break;
        shared_prefix_chain.push_back(common_hash);
        shared_chain_length++;
    }
}

void ChunkedHashTree_RL::update_chain_on_activate(uint32_t request_id)
{
    auto request = request_hashes.find(request_id);
    if (request == request_hashes.end()) return;
    const auto& hashes = request->second;

    if (active_requests.size() == 1) {
        shared_prefix_chain = hashes;
        shared_chain_length = hashes.size();
        return;
    }

    uint32_t low = 0;
    uint32_t high = std::min(shared_chain_length, (uint32_t)hashes.size());
    while (low < high) {
        uint32_t middle = low + (high - low + 1) / 2;
        if (hashes[middle - 1] == shared_prefix_chain[middle - 1]) {
            low = middle;
        } else {
            high = middle - 1;
        }
    }
    shared_chain_length = low;
    shared_prefix_chain.resize(shared_chain_length);
}

void ChunkedHashTree_RL::update_chain_on_finish(uint32_t request_id)
{
    if (active_requests.empty()) {
        shared_prefix_chain.clear();
        shared_chain_length = 0;
        return;
    }
    if (active_requests.size() == 1) {
        recompute_shared_chain();
        return;
    }
    if (!request_hashes.count(request_id)) return;

    uint32_t max_possible = UINT32_MAX;
    for (auto active_id : active_requests) {
        auto active = request_hashes.find(active_id);
        if (active != request_hashes.end()) {
            max_possible = std::min(
                max_possible,
                (uint32_t)active->second.size());
        }
    }

    for (
        uint32_t level = shared_chain_length;
        level < max_possible;
        ++level) {
        uint64_t common_hash = 0;
        bool first = true;
        bool all_match = true;
        for (auto active_id : active_requests) {
            auto active = request_hashes.find(active_id);
            if (
                active == request_hashes.end()
                || level >= active->second.size()) {
                all_match = false;
                break;
            }
            uint64_t hash = active->second[level];
            if (first) {
                common_hash = hash;
                first = false;
            } else if (hash != common_hash) {
                all_match = false;
                break;
            }
        }
        if (!all_match) break;
        shared_prefix_chain.push_back(common_hash);
        shared_chain_length++;
    }
}

uint32_t ChunkedHashTree_RL::compute_chain_length_with_request(
    uint32_t request_id)
{
    auto request = request_hashes.find(request_id);
    if (request == request_hashes.end()) return shared_chain_length;
    const auto& hashes = request->second;
    if (active_requests.empty()) return hashes.size();

    uint32_t matching_length = 0;
    for (
        uint32_t level = 0;
        level < shared_chain_length && level < hashes.size();
        ++level) {
        if (hashes[level] != shared_prefix_chain[level]) break;
        matching_length++;
    }
    return matching_length;
}

uint32_t ChunkedHashTree_RL::count_requests_sharing_prefix(
    uint32_t prefix_length)
{
    if (prefix_length == 0 || prefix_length > shared_chain_length) return 0;
    uint32_t last_level = prefix_length - 1;
    uint64_t key = pack_key(last_level, shared_prefix_chain[last_level]);
    auto requests = hash_to_waiting.find(key);
    if (requests == hash_to_waiting.end()) return 0;

    uint32_t count = 0;
    for (auto request_id : requests->second) {
        if (!active_requests.count(request_id)) count++;
    }
    return count;
}

std::vector<uint64_t>
ChunkedHashTree_RL::compute_hashes(const std::vector<uint32_t>& tokens)
{
    std::vector<uint64_t> hashes;
    hashes.reserve((tokens.size() + chunk_size - 1) / chunk_size);
    XXH64_state_t* state = XXH64_createState();
    XXH64_reset(state, 0);
    for (size_t index = 0; index < tokens.size(); ++index) {
        XXH64_update(state, &tokens[index], sizeof(uint32_t));
        if (
            (index + 1) % chunk_size == 0
            || index + 1 == tokens.size()) {
            hashes.push_back(XXH64_digest(state));
        }
    }
    XXH64_freeState(state);
    return hashes;
}

uint32_t ChunkedHashTree_RL::insert(
    uint32_t request_id,
    const std::vector<uint32_t>& tokens)
{
    if (request_hashes.count(request_id)) return 0;
    auto hashes = compute_hashes(tokens);
    request_hashes[request_id] = hashes;

    int missing = 0;
    for (uint32_t level = 0; level < hashes.size(); ++level) {
        uint64_t key = pack_key(level, hashes[level]);
        if (!working_set.count(key)) missing++;
        auto& waiting = hash_to_waiting[key];
        if (waiting.empty()) waiting.reserve(16);
        waiting.insert(request_id);
    }
    missing_cache[request_id] = missing;
    waiting_heap.push({missing, request_id});
    invalidate_cache();
    return tokens.size();
}

std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>
ChunkedHashTree_RL::find_best_request()
{
    if (cache_valid && cached_best_request.has_value()) {
        return cached_best_request.value();
    }
    while (!waiting_heap.empty()) {
        auto top = waiting_heap.top();
        waiting_heap.pop();
        if (active_requests.count(top.request_id)) continue;
        auto missing = missing_cache.find(top.request_id);
        if (
            missing == missing_cache.end()
            || missing->second != top.missing) {
            continue;
        }

        uint32_t chunks_before = shared_chain_length;
        uint32_t chunks_after = compute_chain_length_with_request(top.request_id);
        uint32_t requests_waiting = count_requests_sharing_prefix(chunks_after);
        cached_best_request = std::make_tuple(
            top.request_id,
            chunks_before,
            chunks_after,
            requests_waiting);
        cache_valid = true;
        waiting_heap.push(top);
        return cached_best_request.value();
    }

    cached_best_request = std::make_tuple(0, 0, 0, 0);
    cache_valid = true;
    return cached_best_request.value();
}

std::pair<uint32_t, uint32_t>
ChunkedHashTree_RL::activate_request(uint32_t request_id)
{
    if (
        active_requests.count(request_id)
        || !request_hashes.count(request_id)) {
        return {0, 0};
    }

    active_requests.insert(request_id);
    auto& hashes = request_hashes[request_id];
    uint32_t added = 0;
    for (uint32_t level = 0; level < hashes.size(); ++level) {
        uint64_t key = pack_key(level, hashes[level]);
        if (hash_ref_counts[key]++ == 0) {
            working_set.insert(key);
            auto waiting = hash_to_waiting.find(key);
            if (waiting != hash_to_waiting.end()) {
                for (auto waiting_id : waiting->second) {
                    if (
                        active_requests.count(waiting_id)
                        || !missing_cache.count(waiting_id)) {
                        continue;
                    }
                    auto& count = missing_cache[waiting_id];
                    if (count > 0) {
                        count--;
                        waiting_heap.push({count, waiting_id});
                    }
                }
            }
            added++;
        }
    }
    missing_cache.erase(request_id);
    update_chain_on_activate(request_id);
    invalidate_cache();
    return {added, (uint32_t)hashes.size()};
}

void ChunkedHashTree_RL::remove(uint32_t request_id)
{
    auto request = request_hashes.find(request_id);
    if (request != request_hashes.end()) {
        for (uint32_t level = 0; level < request->second.size(); ++level) {
            uint64_t key = pack_key(level, request->second[level]);
            auto waiting = hash_to_waiting.find(key);
            if (waiting != hash_to_waiting.end()) {
                waiting->second.erase(request_id);
                if (waiting->second.empty()) hash_to_waiting.erase(waiting);
            }
        }
    }
    request_hashes.erase(request_id);
    missing_cache.erase(request_id);
    invalidate_cache();
}

std::pair<uint32_t, uint32_t>
ChunkedHashTree_RL::finish_request(uint32_t request_id)
{
    if (!active_requests.erase(request_id)) return {0, 0};
    auto request = request_hashes.find(request_id);
    if (request == request_hashes.end()) return {0, 0};

    auto& hashes = request->second;
    uint32_t evicted = 0;
    for (uint32_t level = 0; level < hashes.size(); ++level) {
        uint64_t key = pack_key(level, hashes[level]);
        if (--hash_ref_counts[key] == 0) {
            working_set.erase(key);
            hash_ref_counts.erase(key);
            auto waiting = hash_to_waiting.find(key);
            if (waiting != hash_to_waiting.end()) {
                for (auto waiting_id : waiting->second) {
                    if (
                        active_requests.count(waiting_id)
                        || !missing_cache.count(waiting_id)) {
                        continue;
                    }
                    missing_cache[waiting_id]++;
                    waiting_heap.push(
                        {missing_cache[waiting_id], waiting_id});
                }
            }
            evicted++;
        }
    }
    update_chain_on_finish(request_id);
    remove(request_id);
    invalidate_cache();
    return {evicted, (uint32_t)hashes.size()};
}
