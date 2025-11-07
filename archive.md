---
layout: default
title: Archive
permalink: /archive/
---

# Archive

<div class="archive-list">
{% assign postsByYear = site.posts | group_by_exp:"post", "post.date | date: '%Y'" %}
{% for year in postsByYear %}
  <h2 class="archive-year">{{ year.name }}</h2>
  {% for post in year.items %}
    <div class="archive-item">
      <span class="archive-date">{{ post.date | date: "%b %-d" }}</span>
      <span class="archive-link">
        <a href="{{ site.baseurl }}{{ post.url }}">{{ post.title }}</a>
      </span>
    </div>
  {% endfor %}
{% endfor %}
</div>
