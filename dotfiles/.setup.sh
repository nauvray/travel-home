#!/bin/bash

if [ ! -d ~/code/travel-home ]; then
  mkdir ~/code/travel-home
fi

git clone git@github.com:nauvray/travel-home.git ~/code/travel-home

echo 'alias th="cd ~/code/travel-home"' >> ~/.aliases

cd ~/code/travel-home

echo 'Ready to go ...'
