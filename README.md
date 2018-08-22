# Felicity-Kings-of-ML
Solution for machine learning competition hosted by Analytics Vidhya.

### Requierments:
- numpy
- pandas
- sklearn
- matplotlib
- seaborn
- keras

### Solution files flow:
- data_making.ipynb
- hero_encoding.ipynb
- models.ipynb
- player_encoding.ipynb

### Solution:
- Here I have used keras autoencodders to decrease features for Heroes and Players.
- Then these encoded features and some other features are used to create a predictive model

<h2>Problem Statement</h2>

<p><p style="text-align: justify;"><a href="https://en.wikipedia.org/wiki/Dota_2">Dota2&nbsp;</a>is a<a href="https://en.wikipedia.org/wiki/Free-to-play"> free-to-play</a><a href="https://en.wikipedia.org/wiki/Multiplayer_online_battle_arena"> multiplayer online battle arena</a> (MOBA) video game. <em>Dota 2</em> is played in matches between two teams of five players, with each team occupying and defending their own separate base on the<a href="https://en.wikipedia.org/wiki/Level_(video_gaming)">&nbsp;map</a>. Each of the ten players independentlycontrols a powerful character, known as a "hero" (which they choose at the start of the match), who all have unique <a href="https://en.wikipedia.org/wiki/Skill_(role-playing_games)">abilities</a> and differing styles of play. During a match, players collect<a href="https://en.wikipedia.org/wiki/Experience_point">&nbsp;experience points</a> and<a href="https://en.wikipedia.org/wiki/Item_(gaming)">&nbsp;items</a> for their heroes to successful <a href="https://en.wikipedia.org/wiki/Player_versus_player">battle with the opposing team's </a><a href="https://en.wikipedia.org/wiki/Player_versus_player">heroes</a>, who attempt to do the same to them. A team wins by being the first to destroy a large structure located in the opposing team's base, called the "Ancient".</p>
<p><br /><br /></p>
<p style="text-align: justify;">You&rsquo;re given dataset of professional Dota players and their most frequent 10 heroes. The data also includes details about the heros (Kind of Hero (nuker, initiator and so on), their base attack, strength, movement speed). Here both train and test dataset is divided into two dataset(train9.csv &amp; train1.csv and test9.csv &amp; test1.csv).</p>
<p>&nbsp;</p>
<p>train9.csv and train1.csv contain the user performance for their most frequent 9 heroes and 10th hero respectively. Both train9.csv and train1.csv have below fields.</p>
<p>&nbsp; &nbsp;</p>
<table style="border: none; border-collapse: collapse;">
<tbody>
<tr style="height: 0pt;">
<td style="vertical-align: top; padding: 5pt 5pt 5pt 5pt; border: solid #000000 1pt;">
<p style="line-height: 1.2; margin-top: 0pt; margin-bottom: 0pt;"><strong><span style="font-size: 11pt; font-family: Arial; color: #000000; background-color: transparent; font-variant: normal; text-decoration: none; vertical-align: baseline; white-space: pre;">Feature</span></strong></p>
</td>
<td style="vertical-align: top; padding: 5pt 5pt 5pt 5pt; border: solid #000000 1pt;">
<p style="line-height: 1.2; margin-top: 0pt; margin-bottom: 0pt;"><strong><span style="font-size: 11pt; font-family: Arial; color: #000000; background-color: transparent; font-variant: normal; text-decoration: none; vertical-align: baseline; white-space: pre;">Description</span></strong></p>
</td>
</tr>
<tr style="height: 0pt;">
<td style="vertical-align: top; padding: 5pt 5pt 5pt 5pt; border: solid #000000 1pt;">
<p style="line-height: 1.2; margin-top: 0pt; margin-bottom: 0pt;"><span style="font-size: 11pt; font-family: Arial; color: #000000; background-color: transparent; font-weight: 400; font-variant: normal; text-decoration: none; vertical-align: baseline; white-space: pre;">user_id</span></p>
</td>
<td style="vertical-align: top; padding: 5pt 5pt 5pt 5pt; border: solid #000000 1pt;">
<p style="line-height: 1.2; margin-top: 0pt; margin-bottom: 0pt;"><span style="font-size: 11pt; font-family: Arial; color: #000000; background-color: transparent; font-weight: 400; font-variant: normal; text-decoration: none; vertical-align: baseline; white-space: pre;">The id of the user</span></p>
</td>
</tr>
<tr style="height: 0pt;">
<td style="vertical-align: top; padding: 5pt 5pt 5pt 5pt; border: solid #000000 1pt;">
<p style="line-height: 1.2; margin-top: 0pt; margin-bottom: 0pt;"><span style="font-size: 11pt; font-family: Arial; color: #000000; background-color: transparent; font-weight: 400; font-variant: normal; text-decoration: none; vertical-align: baseline; white-space: pre;">hero_id</span></p>
</td>
<td style="vertical-align: top; padding: 5pt 5pt 5pt 5pt; border: solid #000000 1pt;">
<p style="line-height: 1.2; margin-top: 0pt; margin-bottom: 0pt;"><span style="font-size: 11pt; font-family: Arial; color: #000000; background-color: transparent; font-weight: 400; font-variant: normal; text-decoration: none; vertical-align: baseline; white-space: pre;">The id of the hero the player played with</span></p>
</td>
</tr>
<tr style="height: 0pt;">
<td style="vertical-align: top; padding: 5pt 5pt 5pt 5pt; border: solid #000000 1pt;">
<p style="line-height: 1.2; margin-top: 0pt; margin-bottom: 0pt;"><span style="font-size: 11pt; font-family: Arial; color: #000000; background-color: transparent; font-weight: 400; font-variant: normal; text-decoration: none; vertical-align: baseline; white-space: pre;">id</span></p>
</td>
<td style="vertical-align: top; padding: 5pt 5pt 5pt 5pt; border: solid #000000 1pt;">
<p style="line-height: 1.2; margin-top: 0pt; margin-bottom: 0pt;"><span style="font-size: 11pt; font-family: Arial; color: #000000; background-color: transparent; font-weight: 400; font-variant: normal; text-decoration: none; vertical-align: baseline; white-space: pre;">Unique id</span></p>
</td>
</tr>
<tr style="height: 0pt;">
<td style="vertical-align: top; padding: 5pt 5pt 5pt 5pt; border: solid #000000 1pt;">
<p style="line-height: 1.2; margin-top: 0pt; margin-bottom: 0pt;"><span style="font-size: 11pt; font-family: Arial; color: #000000; background-color: transparent; font-weight: 400; font-variant: normal; text-decoration: none; vertical-align: baseline; white-space: pre;">num_games</span></p>
</td>
<td style="vertical-align: top; padding: 5pt 5pt 5pt 5pt; border: solid #000000 1pt;">
<p style="line-height: 1.2; margin-top: 0pt; margin-bottom: 0pt;"><span style="font-size: 11pt; font-family: Arial; color: #000000; background-color: transparent; font-weight: 400; font-variant: normal; text-decoration: none; vertical-align: baseline; white-space: pre;">The number of games the player played with that hero</span></p>
</td>
</tr>
<tr style="height: 0pt;">
<td style="vertical-align: top; padding: 5pt 5pt 5pt 5pt; border: solid #000000 1pt;">
<p style="line-height: 1.2; margin-top: 0pt; margin-bottom: 0pt;"><span style="font-size: 11pt; font-family: Arial; color: #000000; background-color: transparent; font-weight: 400; font-variant: normal; text-decoration: none; vertical-align: baseline; white-space: pre;">num_wins</span></p>
</td>
<td style="vertical-align: top; padding: 5pt 5pt 5pt 5pt; border: solid #000000 1pt;">
<p style="line-height: 1.2; margin-top: 0pt; margin-bottom: 0pt;"><span style="font-size: 11pt; font-family: Arial; color: #000000; background-color: transparent; font-weight: 400; font-variant: normal; text-decoration: none; vertical-align: baseline; white-space: pre;">Number of games the player won with this particular hero</span></p>
</td>
</tr>
<tr style="height: 0pt;">
<td style="vertical-align: top; padding: 5pt 5pt 5pt 5pt; border: solid #000000 1pt;">
<p style="line-height: 1.2; margin-top: 0pt; margin-bottom: 0pt;"><span style="font-size: 11pt; font-family: Arial; color: #000000; background-color: transparent; font-weight: 400; font-variant: normal; text-decoration: none; vertical-align: baseline; white-space: pre;">kda_ratio (target)</span></p>
</td>
<td style="vertical-align: top; padding: 5pt 5pt 5pt 5pt; border: solid #000000 1pt;">
<p style="line-height: 1.2; margin-top: 0pt; margin-bottom: 0pt;"><span style="font-size: 11pt; font-family: Arial; color: #000000; background-color: transparent; font-weight: 400; font-variant: normal; text-decoration: none; vertical-align: baseline; white-space: pre;">((Kills + Assists)*1000/Deaths) </span></p>
<p style="line-height: 1.2; margin-top: 0pt; margin-bottom: 0pt;"><span style="font-size: 11pt; font-family: Arial; color: #000000; background-color: transparent; font-weight: 400; font-variant: normal; text-decoration: none; vertical-align: baseline; white-space: pre;">Ratio: where kill, assists and deaths are average values per match for that hero</span></p>
</td>
</tr>
</tbody>
</table>
<p><br /><br /></p>
<p style="text-align: justify;">test9.csv contain the different set of user (different from training set) performance for their most frequent 9 heroes. test9.csv has similar fields as train9.csv. Now, <strong>t</strong><strong>he aim is to predict the performance (kda_ratio) of the same set of users (test users) with the 10th hero which is test1.csv.</strong></p>
<p><br /><br /></p>
<table style="border: none; border-collapse: collapse;">
<tbody>
<tr style="height: 0pt;">
<td style="vertical-align: top; padding: 5pt 5pt 5pt 5pt; border: solid #000000 1pt;">
<p style="line-height: 1.2; margin-top: 0pt; margin-bottom: 0pt;"><strong><span style="font-size: 11pt; font-family: Arial; color: #000000; background-color: transparent; font-variant: normal; text-decoration: none; vertical-align: baseline; white-space: pre;">Feature</span></strong></p>
</td>
<td style="vertical-align: top; padding: 5pt 5pt 5pt 5pt; border: solid #000000 1pt;">
<p style="line-height: 1.2; margin-top: 0pt; margin-bottom: 0pt;"><strong><span style="font-size: 11pt; font-family: Arial; color: #000000; background-color: transparent; font-variant: normal; text-decoration: none; vertical-align: baseline; white-space: pre;">Description</span></strong></p>
</td>
</tr>
<tr style="height: 0pt;">
<td style="vertical-align: top; padding: 5pt 5pt 5pt 5pt; border: solid #000000 1pt;">
<p style="line-height: 1.2; margin-top: 0pt; margin-bottom: 0pt;"><span style="font-size: 11pt; font-family: Arial; color: #000000; background-color: transparent; font-weight: 400; font-variant: normal; text-decoration: none; vertical-align: baseline; white-space: pre;">user_id</span></p>
</td>
<td style="vertical-align: top; padding: 5pt 5pt 5pt 5pt; border: solid #000000 1pt;">
<p style="line-height: 1.2; margin-top: 0pt; margin-bottom: 0pt;"><span style="font-size: 11pt; font-family: Arial; color: #000000; background-color: transparent; font-weight: 400; font-variant: normal; text-decoration: none; vertical-align: baseline; white-space: pre;">The id of the user</span></p>
</td>
</tr>
<tr style="height: 0pt;">
<td style="vertical-align: top; padding: 5pt 5pt 5pt 5pt; border: solid #000000 1pt;">
<p style="line-height: 1.2; margin-top: 0pt; margin-bottom: 0pt;"><span style="font-size: 11pt; font-family: Arial; color: #000000; background-color: transparent; font-weight: 400; font-variant: normal; text-decoration: none; vertical-align: baseline; white-space: pre;">hero_id</span></p>
</td>
<td style="vertical-align: top; padding: 5pt 5pt 5pt 5pt; border: solid #000000 1pt;">
<p style="line-height: 1.2; margin-top: 0pt; margin-bottom: 0pt;"><span style="font-size: 11pt; font-family: Arial; color: #000000; background-color: transparent; font-weight: 400; font-variant: normal; text-decoration: none; vertical-align: baseline; white-space: pre;">The id of the hero (of which the kda_ratio has to be predicted)</span></p>
</td>
</tr>
<tr style="height: 0pt;">
<td style="vertical-align: top; padding: 5pt 5pt 5pt 5pt; border: solid #000000 1pt;">
<p style="line-height: 1.2; margin-top: 0pt; margin-bottom: 0pt;"><span style="font-size: 11pt; font-family: Arial; color: #000000; background-color: transparent; font-weight: 400; font-variant: normal; text-decoration: none; vertical-align: baseline; white-space: pre;">id</span></p>
</td>
<td style="vertical-align: top; padding: 5pt 5pt 5pt 5pt; border: solid #000000 1pt;">
<p style="line-height: 1.2; margin-top: 0pt; margin-bottom: 0pt;"><span style="font-size: 11pt; font-family: Arial; color: #000000; background-color: transparent; font-weight: 400; font-variant: normal; text-decoration: none; vertical-align: baseline; white-space: pre;">Unique id</span></p>
</td>
</tr>
<tr style="height: 0pt;">
<td style="vertical-align: top; padding: 5pt 5pt 5pt 5pt; border: solid #000000 1pt;">
<p style="line-height: 1.2; margin-top: 0pt; margin-bottom: 0pt;"><span style="font-size: 11pt; font-family: Arial; color: #000000; background-color: transparent; font-weight: 400; font-variant: normal; text-decoration: none; vertical-align: baseline; white-space: pre;">num_games</span></p>
</td>
<td style="vertical-align: top; padding: 5pt 5pt 5pt 5pt; border: solid #000000 1pt;">
<p style="line-height: 1.2; margin-top: 0pt; margin-bottom: 0pt;"><span style="font-size: 11pt; font-family: Arial; color: #000000; background-color: transparent; font-weight: 400; font-variant: normal; text-decoration: none; vertical-align: baseline; white-space: pre;">The total number of games the player played with this hero</span></p>
</td>
</tr>
</tbody>
</table>
<p>&nbsp;<br /><br /></p>
<p style="line-height: 1.38; margin-top: 0pt; margin-bottom: 0pt;"><span style="font-size: 11pt; font-family: Arial; color: #000000; background-color: transparent; font-weight: 400; font-variant: normal; text-decoration: none; vertical-align: baseline; white-space: pre;">We also have "hero_data.csv" which contains information about heros.</span></p>
<table style="border: none; border-collapse: collapse;">
<tbody>
<tr style="height: 0pt;">
<td style="vertical-align: top; padding: 5pt 5pt 5pt 5pt; border: solid #000000 1pt;">
<p style="line-height: 1.2; margin-top: 0pt; margin-bottom: 0pt;"><strong><span style="font-size: 11pt; font-family: Arial; color: #000000; background-color: transparent; font-variant: normal; text-decoration: none; vertical-align: baseline; white-space: pre;">Feature</span></strong></p>
</td>
<td style="vertical-align: top; padding: 5pt 5pt 5pt 5pt; border: solid #000000 1pt;">
<p style="line-height: 1.2; margin-top: 0pt; margin-bottom: 0pt;"><strong><span style="font-size: 11pt; font-family: Arial; color: #000000; background-color: transparent; font-variant: normal; text-decoration: none; vertical-align: baseline; white-space: pre;">Description</span></strong></p>
</td>
</tr>
<tr style="height: 0pt;">
<td style="vertical-align: top; padding: 5pt 5pt 5pt 5pt; border: solid #000000 1pt;">
<p style="line-height: 1.2; margin-top: 0pt; margin-bottom: 0pt;"><span style="font-size: 11pt; font-family: Arial; color: #000000; background-color: transparent; font-weight: 400; font-variant: normal; text-decoration: none; vertical-align: baseline; white-space: pre;">hero_id</span></p>
</td>
<td style="vertical-align: top; padding: 5pt 5pt 5pt 5pt; border: solid #000000 1pt;">
<p style="line-height: 1.2; margin-top: 0pt; margin-bottom: 0pt;"><span style="font-size: 11pt; font-family: Arial; color: #000000; background-color: transparent; font-weight: 400; font-variant: normal; text-decoration: none; vertical-align: baseline; white-space: pre;">Id of the hero</span></p>
</td>
</tr>
<tr style="height: 0pt;">
<td style="vertical-align: top; padding: 5pt 5pt 5pt 5pt; border: solid #000000 1pt;">
<p style="line-height: 1.2; margin-top: 0pt; margin-bottom: 0pt;"><span style="font-size: 11pt; font-family: Arial; color: #000000; background-color: transparent; font-weight: 400; font-variant: normal; text-decoration: none; vertical-align: baseline; white-space: pre;">primary_attr</span></p>
</td>
<td style="vertical-align: top; padding: 5pt 5pt 5pt 5pt; border: solid #000000 1pt;">
<p style="line-height: 1.2; margin-top: 0pt; margin-bottom: 0pt;"><span style="font-size: 11pt; font-family: Arial; color: #000000; background-color: transparent; font-weight: 400; font-variant: normal; text-decoration: none; vertical-align: baseline; white-space: pre;">A string denoting what the primary attribute of the hero is</span></p>
<p style="line-height: 1.2; margin-top: 0pt; margin-bottom: 0pt;"><span style="font-size: 11pt; font-family: Arial; color: #000000; background-color: transparent; font-weight: 400; font-variant: normal; text-decoration: none; vertical-align: baseline; white-space: pre;"> (int- initiator, agi- agility, str- strength and so on)</span></p>
</td>
</tr>
<tr style="height: 0pt;">
<td style="vertical-align: top; padding: 5pt 5pt 5pt 5pt; border: solid #000000 1pt;">
<p style="line-height: 1.2; margin-top: 0pt; margin-bottom: 0pt;"><span style="font-size: 11pt; font-family: Arial; color: #000000; background-color: transparent; font-weight: 400; font-variant: normal; text-decoration: none; vertical-align: baseline; white-space: pre;">attack_type</span></p>
</td>
<td style="vertical-align: top; padding: 5pt 5pt 5pt 5pt; border: solid #000000 1pt;">
<p style="line-height: 1.2; margin-top: 0pt; margin-bottom: 0pt;"><span style="font-size: 11pt; font-family: Arial; color: #000000; background-color: transparent; font-weight: 400; font-variant: normal; text-decoration: none; vertical-align: baseline; white-space: pre;">String, :&rdquo;Melee&ldquo; or &ldquo;Ranged&rdquo;</span></p>
</td>
</tr>
<tr style="height: 0pt;">
<td style="vertical-align: top; padding: 5pt 5pt 5pt 5pt; border: solid #000000 1pt;">
<p style="line-height: 1.2; margin-top: 0pt; margin-bottom: 0pt;"><span style="font-size: 11pt; font-family: Arial; color: #000000; background-color: transparent; font-weight: 400; font-variant: normal; text-decoration: none; vertical-align: baseline; white-space: pre;">roles</span></p>
</td>
<td style="vertical-align: top; padding: 5pt 5pt 5pt 5pt; border: solid #000000 1pt;">
<p style="line-height: 1.2; margin-top: 0pt; margin-bottom: 0pt;"><span style="font-size: 11pt; font-family: Arial; color: #000000; background-color: transparent; font-weight: 400; font-variant: normal; text-decoration: none; vertical-align: baseline; white-space: pre;">An array of strings which have roles of heroes</span></p>
<p style="line-height: 1.2; margin-top: 0pt; margin-bottom: 0pt;"><span style="font-size: 11pt; font-family: Arial; color: #000000; background-color: transparent; font-weight: 400; font-variant: normal; text-decoration: none; vertical-align: baseline; white-space: pre;"> (eg Support, Disabler, Nuker, etc.)</span></p>
</td>
</tr>
<tr style="height: 0pt;">
<td style="vertical-align: top; padding: 5pt 5pt 5pt 5pt; border: solid #000000 1pt;">
<p style="line-height: 1.2; margin-top: 0pt; margin-bottom: 0pt;"><span style="font-size: 11pt; font-family: Arial; color: #000000; background-color: transparent; font-weight: 400; font-variant: normal; text-decoration: none; vertical-align: baseline; white-space: pre;">base_health</span></p>
</td>
<td style="vertical-align: top; padding: 5pt 5pt 5pt 5pt; border: solid #000000 1pt;">
<p style="line-height: 1.2; margin-top: 0pt; margin-bottom: 0pt;"><span style="font-size: 11pt; font-family: Arial; color: #000000; background-color: transparent; font-weight: 400; font-variant: normal; text-decoration: none; vertical-align: baseline; white-space: pre;">The basic health the hero starts with</span></p>
</td>
</tr>
<tr style="height: 0pt;">
<td style="vertical-align: top; padding: 5pt 5pt 5pt 5pt; border: solid #000000 1pt;">
<p style="line-height: 1.2; margin-top: 0pt; margin-bottom: 0pt;"><span style="font-size: 11pt; font-family: Arial; color: #000000; background-color: transparent; font-weight: 400; font-variant: normal; text-decoration: none; vertical-align: baseline; white-space: pre;">base_health_regen,base_mana,base_mana_regen,</span></p>
<p style="line-height: 1.2; margin-top: 0pt; margin-bottom: 0pt;"><span style="font-size: 11pt; font-family: Arial; color: #000000; background-color: transparent; font-weight: 400; font-variant: normal; text-decoration: none; vertical-align: baseline; white-space: pre;">base_armor,base_magic_restistance,</span></p>
<p style="line-height: 1.2; margin-top: 0pt; margin-bottom: 0pt;"><span style="font-size: 11pt; font-family: Arial; color: #000000; background-color: transparent; font-weight: 400; font-variant: normal; text-decoration: none; vertical-align: baseline; white-space: pre;">base_attack_min,base_attack_max,base_strength,</span></p>
<p style="line-height: 1.2; margin-top: 0pt; margin-bottom: 0pt;"><span style="font-size: 11pt; font-family: Arial; color: #000000; background-color: transparent; font-weight: 400; font-variant: normal; text-decoration: none; vertical-align: baseline; white-space: pre;">base_agility,base_intelligence,strength_gain,agility_gain,intelligence_gain,</span></p>
<p style="line-height: 1.2; margin-top: 0pt; margin-bottom: 0pt;"><span style="font-size: 11pt; font-family: Arial; color: #000000; background-color: transparent; font-weight: 400; font-variant: normal; text-decoration: none; vertical-align: baseline; white-space: pre;">attack_range,projectile_speed,attack_rate,move_speed,turn_rate</span></p>
</td>
<td style="vertical-align: top; padding: 5pt 5pt 5pt 5pt; border: solid #000000 1pt;">
<p style="line-height: 1.2; margin-top: 0pt; margin-bottom: 0pt;"><span style="font-size: 11pt; font-family: Arial; color: #000000; background-color: transparent; font-weight: 400; font-variant: normal; text-decoration: none; vertical-align: baseline; white-space: pre;">These are the basic stats the heroes start with </span></p>
<p style="line-height: 1.2; margin-top: 0pt; margin-bottom: 0pt;"><span style="font-size: 11pt; font-family: Arial; color: #000000; background-color: transparent; font-weight: 400; font-variant: normal; text-decoration: none; vertical-align: baseline; white-space: pre;">(some remain same throughout the game)</span></p>
</td>
</tr>
</tbody>
</table>
<p>&nbsp;</p>
<h2>Evaluation Metric</h2>
<p>The predictions will be evaluated on RMSE.</p>
<p>The public private split is 40:60</p> </p>
