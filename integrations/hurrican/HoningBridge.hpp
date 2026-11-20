// Local synchronous control hooks for the pinned Hurrican SDL2 source.
#pragma once
#include <bitset>
#include "Player.hpp"

bool HoningEnabled();
bool HoningBeforeTick();
void HoningAfterTick();
std::bitset<MAX_AKTIONEN> HoningActions();
void HoningCapture(SDL_Window *window);
